"""Create a Parquet inventory of files in a directory tree."""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def collect_file_metadata(source: Path, output: Path) -> list[dict[str, Any]]:
    """Collect metadata for every file below source, excluding the output file."""
    output_resolved = output.resolve()
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
            if file_path.resolve() == output_resolved:
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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Recursively inventory files and write their metadata to Parquet."
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
        help="Output Parquet file (default: ./file_inventory.parquet)",
    )
    args = parser.parse_args()

    source = args.source.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not source.exists():
        parser.error(f"source directory does not exist: {source}")
    if not source.is_dir():
        parser.error(f"source is not a directory: {source}")

    try:
        rows = collect_file_metadata(source, output)
        write_parquet(rows, output)
    except (OSError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    print(f"Wrote {len(rows)} file records to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
