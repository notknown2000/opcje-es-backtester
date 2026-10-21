from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
LOCAL_DEPS = PROJECT_ROOT / ".deps"
if LOCAL_DEPS.exists():
    sys.path.insert(0, str(LOCAL_DEPS))

import pyarrow as pa
import pyarrow.csv as csv
import pyarrow.parquet as pq


INPUT_GLOB = "spx_eod_*.txt"


def clean_column_name(name: str) -> str:
    return name.strip().replace("[", "").replace("]", "")


def build_convert_options() -> csv.ConvertOptions:
    return csv.ConvertOptions(
        column_types={
            "QUOTE_UNIXTIME": pa.int64(),
            "QUOTE_READTIME": pa.string(),
            "QUOTE_DATE": pa.date32(),
            "QUOTE_TIME_HOURS": pa.float64(),
            "UNDERLYING_LAST": pa.float64(),
            "EXPIRE_DATE": pa.date32(),
            "EXPIRE_UNIX": pa.int64(),
            "DTE": pa.float64(),
            "C_DELTA": pa.float64(),
            "C_GAMMA": pa.float64(),
            "C_VEGA": pa.float64(),
            "C_THETA": pa.float64(),
            "C_RHO": pa.float64(),
            "C_IV": pa.float64(),
            "C_VOLUME": pa.float64(),
            "C_LAST": pa.float64(),
            "C_SIZE": pa.string(),
            "C_BID": pa.float64(),
            "C_ASK": pa.float64(),
            "STRIKE": pa.float64(),
            "P_BID": pa.float64(),
            "P_ASK": pa.float64(),
            "P_SIZE": pa.string(),
            "P_LAST": pa.float64(),
            "P_DELTA": pa.float64(),
            "P_GAMMA": pa.float64(),
            "P_VEGA": pa.float64(),
            "P_THETA": pa.float64(),
            "P_RHO": pa.float64(),
            "P_IV": pa.float64(),
            "P_VOLUME": pa.float64(),
            "STRIKE_DISTANCE": pa.float64(),
            "STRIKE_DISTANCE_PCT": pa.float64(),
        },
        strings_can_be_null=True,
        null_values=["", "null", "NULL", "nan", "NaN"],
    )


def convert_file(source_path: Path, output_dir: Path, compression: str) -> tuple[Path, int]:
    target_path = output_dir / f"{source_path.stem}.parquet"
    read_options = csv.ReadOptions(use_threads=True)
    parse_options = csv.ParseOptions(delimiter=",", ignore_empty_lines=True)
    convert_options = build_convert_options()

    table = csv.read_csv(
        source_path,
        read_options=read_options,
        parse_options=parse_options,
        convert_options=convert_options,
    )
    table = table.rename_columns([clean_column_name(name) for name in table.column_names])
    pq.write_table(table, target_path, compression=compression)
    return target_path, table.num_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Konwersja plikow SPX EOD TXT do Parquet.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=PROJECT_ROOT / "data",
        help="Katalog z plikami wejsciowymi TXT.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data-parquet",
        help="Katalog docelowy dla plikow Parquet.",
    )
    parser.add_argument(
        "--pattern",
        default=INPUT_GLOB,
        help="Glob plikow do przetworzenia, np. spx_eod_201001.txt lub spx_eod_*.txt.",
    )
    parser.add_argument(
        "--compression",
        default="zstd",
        choices=["none", "snappy", "gzip", "brotli", "lz4", "zstd"],
        help="Kompresja Parquet.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    compression = None if args.compression == "none" else args.compression

    source_files = sorted(input_dir.glob(args.pattern))
    if not source_files:
        print(f"Brak plikow do przetworzenia w {input_dir} dla wzorca {args.pattern}")
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)
    started_at = time.perf_counter()
    total_rows = 0

    print(f"Znaleziono {len(source_files)} plikow do konwersji.")
    for index, source_path in enumerate(source_files, start=1):
        file_started_at = time.perf_counter()
        target_path, row_count = convert_file(source_path, output_dir, compression)
        total_rows += row_count
        elapsed = time.perf_counter() - file_started_at
        print(
            f"[{index}/{len(source_files)}] {source_path.name} -> {target_path.name} "
            f"({row_count} wierszy, {elapsed:.1f}s)"
        )

    total_elapsed = time.perf_counter() - started_at
    print(f"Gotowe. Przetworzono {total_rows} wierszy w {total_elapsed:.1f}s.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
