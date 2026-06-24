"""Convert generated depth CSV training shards to Parquet."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from models.supervised_policy import (  # noqa: E402
    DEFAULT_CSV_DATA_DIR,
    DEFAULT_PARQUET_DATA_DIR,
    TRAINING_ROW_COLUMNS,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert depth_N.csv training shards to depth_N.parquet."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_CSV_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_PARQUET_DATA_DIR)
    parser.add_argument("--chunksize", type=int, default=100_000)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    paths = convert_training_data_to_parquet(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        chunksize=args.chunksize,
        overwrite=args.overwrite,
    )
    for path in paths:
        print(path)


def convert_training_data_to_parquet(
    data_dir: Path | str = DEFAULT_CSV_DATA_DIR,
    output_dir: Path | str | None = None,
    chunksize: int = 100_000,
    overwrite: bool = False,
) -> list[Path]:
    """Convert each depth_N.csv shard to a matching Parquet shard."""

    if chunksize <= 0:
        raise ValueError("chunksize must be positive")

    try:
        import pandas as pd
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as error:
        raise ImportError(
            "CSV to Parquet conversion requires pandas and pyarrow. "
            "Install dependencies with `python -m pip install -r requirements.txt`."
        ) from error

    source_dir = Path(data_dir)
    destination_dir = Path(output_dir) if output_dir is not None else source_dir
    destination_dir.mkdir(parents=True, exist_ok=True)

    csv_paths = sorted(source_dir.glob("depth_*.csv"), key=_depth_sort_key)
    if not csv_paths:
        raise ValueError(f"no depth_*.csv files found in {source_dir}")

    output_paths: list[Path] = []
    for csv_path in csv_paths:
        parquet_path = destination_dir / f"{csv_path.stem}.parquet"
        if parquet_path.exists() and not overwrite:
            output_paths.append(parquet_path)
            continue

        writer = None
        try:
            for chunk in pd.read_csv(
                csv_path,
                chunksize=chunksize,
                usecols=lambda column: column in TRAINING_ROW_COLUMNS,
            ):
                table = pa.Table.from_pandas(chunk, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(parquet_path, table.schema)
                writer.write_table(table)
        finally:
            if writer is not None:
                writer.close()

        output_paths.append(parquet_path)

    return output_paths


def _depth_sort_key(path: Path) -> tuple[int, str]:
    try:
        return int(path.stem.removeprefix("depth_")), path.name
    except ValueError:
        return 0, path.name


if __name__ == "__main__":
    main()
