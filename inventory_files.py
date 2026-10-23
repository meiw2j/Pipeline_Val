"""Create a Parquet inventory of files in a directory tree."""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def collect_file_metadata(
    source: Path, output: Path, additional_outputs: set[Path] | None = None
) -> list[dict[str, Any]]:
    """Collect metadata for every file below source, excluding generated outputs."""
    excluded_outputs = {output.resolve()}
    if additional_outputs:
        excluded_outputs.update(path.resolve() for path in additional_outputs)
    rows: list[dict[str, Any]] = []

    def raise_walk_error(error: OSError) -> None:
        raise error

    for root, directories, filenames in os.walk(
        source, onerror=raise_walk_error, followlinks=False
    ):
        directories.sort()
        filenames.sort()
        for filename in filenames:
            file_path = Path(root) / filename
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

    def raise_walk_error(error: OSError) -> None:
        raise error

    for root, directories, filenames in os.walk(
        source, onerror=raise_walk_error, followlinks=False
    ):
        directories.sort()
        filenames.sort()
        for filename in filenames:
            file_path = Path(root) / filename
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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Write recursive file and Parquet table metadata inventories."
    )
    parser.add_argument(
        "source",
        type=Path,
        help="Directory to scan (for example, /data002/eds/data/prod/raw_files/P2026W38)",
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
        help="Parquet table metadata output (default: ./parquet_table_inventory.parquet)",
    )
    args = parser.parse_args()

    source = args.source.expanduser().resolve()
    output = args.output.expanduser().resolve()
    table_output = args.table_output.expanduser().resolve()
    if not source.exists():
        parser.error(f"source directory does not exist: {source}")
    if not source.is_dir():
        parser.error(f"source is not a directory: {source}")
    if output == table_output:
        parser.error("--output and --table-output must be different files")

    try:
        generated_outputs = {output, table_output}
        rows = collect_file_metadata(source, output, {table_output})
        table_rows = collect_parquet_table_metadata(source, generated_outputs)
        write_parquet(rows, output)
        write_table_metadata(table_rows, table_output)
    except (OSError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    print(f"Wrote {len(rows)} file records to {output}")
    print(f"Wrote {len(table_rows)} Parquet table records to {table_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
