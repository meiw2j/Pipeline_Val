"""Create a Parquet inventory of files in a directory tree."""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


def iter_source_files(source: Path) -> Iterator[Path]:
    """Yield a source file or all files in a directory tree in sorted order."""
    if source.is_file():
        yield source
        return

    def raise_walk_error(error: OSError) -> None:
        raise error

    for root, directories, filenames in os.walk(
        source, onerror=raise_walk_error, followlinks=False
    ):
        directories.sort()
        filenames.sort()
        for filename in filenames:
            yield Path(root) / filename


def collect_file_metadata(
    source: Path, output: Path, additional_outputs: set[Path] | None = None
) -> list[dict[str, Any]]:
    """Collect metadata for the source file or directory tree."""
    excluded_outputs = {output.resolve()}
    if additional_outputs:
        excluded_outputs.update(path.resolve() for path in additional_outputs)
    rows: list[dict[str, Any]] = []

    for file_path in iter_source_files(source):
        if file_path.resolve() in excluded_outputs:
            continue

        stat = file_path.stat()
        rows.append(
            {
                "filename": file_path.name,
                "fullpath": str(file_path.resolve()),
                "filesize": stat.st_size,
                "file extensions": file_path.suffix,
                "modified_time": datetime.fromtimestamp(
                    stat.st_mtime, tz=timezone.utc
                ),
            }
        )

    return rows


def collect_parquet_table_metadata(
    source: Path, excluded_outputs: set[Path]
) -> list[dict[str, Any]]:
    """Collect row, column, size, and modification metadata for Parquet files."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows: list[dict[str, Any]] = []

    for file_path in iter_source_files(source):
        if file_path.suffix.lower() != ".parquet":
            continue
        resolved_path = file_path.resolve()
        if resolved_path in excluded_outputs:
            continue

        try:
            stat = file_path.stat()
            metadata = pq.ParquetFile(file_path).metadata
        except (OSError, pa.ArrowException) as error:
            raise RuntimeError(
                f"Could not read Parquet metadata from {file_path}: {error}"
            ) from error

        rows.append(
            {
                "table_name": file_path.stem,
                "table_size": stat.st_size,
                "NOBS": metadata.num_rows,
                "NVARS": metadata.num_columns,
                "modified_time": datetime.fromtimestamp(
                    stat.st_mtime, tz=timezone.utc
                ),
            }
        )

    return rows


def _fixed_width_bytes(data_type: Any) -> int | None:
    import pyarrow as pa

    if pa.types.is_fixed_size_binary(data_type):
        return data_type.byte_width
    if (
        pa.types.is_boolean(data_type)
        or pa.types.is_integer(data_type)
        or pa.types.is_floating(data_type)
        or pa.types.is_decimal(data_type)
        or pa.types.is_temporal(data_type)
    ):
        return (data_type.bit_width + 7) // 8
    return None


def collect_parquet_column_metadata(
    source: Path, include_details: bool = False
) -> list[dict[str, Any]]:
    """Collect basic column metadata and optionally detailed statistics."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    try:
        parquet_file = pq.ParquetFile(source)
        schema = parquet_file.schema_arrow
        parquet_file.close()
        rows = [
            {
                "Varname": field.name,
                "Vartype": str(field.type),
                "Varlen": _fixed_width_bytes(field.type),
            }
            for field in schema
        ]
        if include_details:
            statistics = _collect_column_statistics_with_duckdb(source, schema)
            for row, stats in zip(rows, statistics):
                row.update(stats)
        return rows
    except (OSError, pa.ArrowException) as error:
        raise RuntimeError(
            f"Could not calculate column metadata from {source}: {error}"
        ) from error


def _collect_column_statistics_with_duckdb(
    source: Path, schema: Any
) -> list[dict[str, Any]]:
    try:
        import duckdb
        import pyarrow as pa
    except ImportError as error:
        raise RuntimeError(
            "Detailed column statistics require DuckDB and PyArrow. Install them with "
            "'python -m pip install -r requirements.txt'."
        ) from error

    expressions = ["count(*) AS __nobs"]
    column_aliases: list[dict[str, str | None]] = []
    for index, field in enumerate(schema):
        column_name = '"' + field.name.replace('"', '""') + '"'
        data_type = field.type
        is_numeric = (
            pa.types.is_integer(data_type)
            or pa.types.is_floating(data_type)
            or pa.types.is_decimal(data_type)
        )
        aliases: dict[str, str | None] = {
            "n_pop": f"__c{index}_n_pop",
            "distinct": None,
            "min": None,
            "max": None,
            "sum": None,
            "varlen": None,
        }
        expressions.append(
            f"count({column_name}) AS {aliases['n_pop']}"
        )

        if is_numeric:
            aliases.update(
                {
                    "min": f"__c{index}_min",
                    "max": f"__c{index}_max",
                    "sum": f"__c{index}_sum",
                }
            )
            expressions.extend(
                [
                    f"min({column_name}) AS {aliases['min']}",
                    f"max({column_name}) AS {aliases['max']}",
                    f"sum({column_name}) AS {aliases['sum']}",
                ]
            )
        else:
            aliases["distinct"] = f"__c{index}_distinct"
            expressions.append(
                f"count(DISTINCT {column_name}) AS {aliases['distinct']}"
            )

        if pa.types.is_string(data_type) or pa.types.is_large_string(data_type):
            aliases["varlen"] = f"__c{index}_varlen"
            expressions.append(
                f"max(strlen({column_name})) AS {aliases['varlen']}"
            )
        elif (
            pa.types.is_binary(data_type)
            or pa.types.is_large_binary(data_type)
            or pa.types.is_fixed_size_binary(data_type)
        ):
            aliases["varlen"] = f"__c{index}_varlen"
            expressions.append(
                f"max(octet_length({column_name})) AS {aliases['varlen']}"
            )
        column_aliases.append(aliases)

    connection = None
    try:
        connection = duckdb.connect(database=":memory:")
        query = (
            "SELECT "
            + ", ".join(expressions)
            + " FROM read_parquet(?)"
        )
        cursor = connection.execute(query, [str(source)])
        aggregate_values = dict(
            zip((description[0] for description in cursor.description), cursor.fetchone())
        )
        nobs = aggregate_values["__nobs"]
        statistics: list[dict[str, Any]] = []
        for aliases in column_aliases:
            n_pop = aggregate_values[aliases["n_pop"]]
            varlen = aggregate_values.get(aliases.get("varlen", ""))
            distinct = aggregate_values.get(aliases.get("distinct", ""))
            minimum = aggregate_values.get(aliases.get("min", ""))
            maximum = aggregate_values.get(aliases.get("max", ""))
            total = aggregate_values.get(aliases.get("sum", ""))
            result: dict[str, Any] = {
                "Varlen": varlen,
                "N_Pop": n_pop,
                "N_Missing": nobs - n_pop,
                "Pop_Rate": n_pop / nobs if nobs else None,
                "N_Distinct": distinct,
                "Min": float(minimum) if minimum is not None else None,
                "Max": float(maximum) if maximum is not None else None,
                "Sum": float(total) if total is not None else None,
            }
            statistics.append(result)
        return statistics
    except duckdb.Error as error:
        raise RuntimeError(
            f"DuckDB could not calculate column statistics from {source}: {error}"
        ) from error
    finally:
        if connection is not None:
            connection.close()


def write_parquet(rows: list[dict[str, Any]], output: Path) -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError(
            "Writing Parquet requires pyarrow. Install it with: "
            "python -m pip install -r requirements.txt"
        ) from error

    schema = pa.schema(
        [
            ("filename", pa.string()),
            ("fullpath", pa.string()),
            ("filesize", pa.int64()),
            ("file extensions", pa.string()),
            ("modified_time", pa.timestamp("us", tz="UTC")),
        ]
    )
    table = pa.Table.from_pylist(rows, schema=schema)
    output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, output)


def write_table_metadata(rows: list[dict[str, Any]], output: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    schema = pa.schema(
        [
            ("table_name", pa.string()),
            ("table_size", pa.int64()),
            ("NOBS", pa.int64()),
            ("NVARS", pa.int64()),
            ("modified_time", pa.timestamp("us", tz="UTC")),
        ]
    )
    table = pa.Table.from_pylist(rows, schema=schema)
    output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, output)


def write_column_metadata(
    rows: list[dict[str, Any]], output: Path, include_details: bool = False
) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    fields = [
        ("Varname", pa.string()),
        ("Vartype", pa.string()),
        ("Varlen", pa.int64()),
    ]
    if include_details:
        fields.extend(
            [
                ("N_Pop", pa.int64()),
                ("N_Missing", pa.int64()),
                ("Pop_Rate", pa.float64()),
                ("N_Distinct", pa.int64()),
                ("Min", pa.float64()),
                ("Max", pa.float64()),
                ("Sum", pa.float64()),
            ]
        )
    schema = pa.schema(fields)
    table = pa.Table.from_pylist(rows, schema=schema)
    output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, output)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Write recursive file and Parquet table metadata inventories."
    )
    parser.add_argument(
        "source",
        type=Path,
        help="Parquet file or directory to scan",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("file_inventory.parquet"),
        help="File inventory output (default: ./file_inventory.parquet)",
    )
    parser.add_argument(
        "--table-output",
        type=Path,
        default=Path("parquet_table_inventory.parquet"),
        help="Parquet table metadata output "
        "(default: ./parquet_table_inventory.parquet)",
    )
    parser.add_argument(
        "--column-output",
        type=Path,
        help="Column metadata output for a single Parquet input "
        "(default: ./parquet_column_inventory.parquet)",
    )
    parser.add_argument(
        "--column-detail-output",
        type=Path,
        help="Detailed column statistics output (requires DuckDB) "
        "(default: ./parquet_column_detail_inventory.parquet)",
    )
    parser.add_argument(
        "--outputs",
        nargs="+",
        choices=("file", "table", "column", "column-detail", "all"),
        help="Outputs to generate: file, table, column, column-detail, or all. "
        "The default/all includes basic column metadata; detailed statistics "
        "require column-detail or an explicit --column-detail-output path "
        "and DuckDB.",
    )
    args = parser.parse_args()

    source = args.source.expanduser().resolve()
    output = args.output.expanduser().resolve()
    table_output = args.table_output.expanduser().resolve()
    column_output = (
        args.column_output.expanduser().resolve()
        if args.column_output
        else Path("parquet_column_inventory.parquet").resolve()
        if source.is_file()
        else None
    )
    column_detail_output = (
        args.column_detail_output.expanduser().resolve()
        if args.column_detail_output
        else Path("parquet_column_detail_inventory.parquet").resolve()
        if source.is_file()
        else None
    )
    if args.outputs is None or args.outputs == ["all"]:
        selected_outputs = {"file", "table"}
        if source.is_file():
            selected_outputs.add("column")
            if args.outputs is None and args.column_detail_output is not None:
                selected_outputs.add("column-detail")
    elif "all" in args.outputs:
        parser.error("'all' cannot be combined with specific output names")
    else:
        selected_outputs = set(args.outputs)

    if not source.exists():
        parser.error(f"source does not exist: {source}")
    if not source.is_dir() and not source.is_file():
        parser.error(f"source is not a regular file or directory: {source}")
    if source.is_file() and source.suffix.lower() != ".parquet":
        parser.error(f"source file is not a Parquet file: {source}")
    if args.column_output is not None and not source.is_file():
        parser.error("--column-output requires a single Parquet file input")
    if args.column_detail_output is not None and not source.is_file():
        parser.error("--column-detail-output requires a single Parquet file input")
    if {"column", "column-detail"} & selected_outputs and not source.is_file():
        parser.error("column outputs require a single Parquet file input")

    output_paths = {
        "file": output,
        "table": table_output,
        "column": column_output,
        "column-detail": column_detail_output,
    }
    selected_paths = [
        output_paths[name] for name in selected_outputs if output_paths[name] is not None
    ]
    if len(set(selected_paths)) != len(selected_paths):
        parser.error("output paths must be different files")
    if source.is_file() and source in selected_paths:
        parser.error("output files must not overwrite the source Parquet file")

    try:
        excluded_outputs = {
            path
            for path in (output, table_output, column_output, column_detail_output)
            if path is not None
        }
        if "file" in selected_outputs:
            rows = collect_file_metadata(source, output, excluded_outputs - {output})
            write_parquet(rows, output)
            print(f"Wrote {len(rows)} file records to {output}")
        if "table" in selected_outputs:
            table_rows = collect_parquet_table_metadata(source, excluded_outputs)
            write_table_metadata(table_rows, table_output)
            print(
                f"Wrote {len(table_rows)} Parquet table records to {table_output}"
            )
        if "column" in selected_outputs:
            column_rows = collect_parquet_column_metadata(source)
            assert column_output is not None
            write_column_metadata(column_rows, column_output)
            print(f"Wrote {len(column_rows)} column records to {column_output}")
        if "column-detail" in selected_outputs:
            detail_rows = collect_parquet_column_metadata(
                source, include_details=True
            )
            assert column_detail_output is not None
            write_column_metadata(detail_rows, column_detail_output, include_details=True)
            print(
                f"Wrote {len(detail_rows)} detailed column records "
                f"to {column_detail_output}"
            )
    except (OSError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
