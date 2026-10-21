from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
LOCAL_DEPS = PROJECT_ROOT / ".deps"
if LOCAL_DEPS.exists():
    sys.path.insert(0, str(LOCAL_DEPS))

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq


DATA_DIR = PROJECT_ROOT / "data-parquet"
ATM_DELTA_MIN = 0.48
ATM_DELTA_MAX = 0.52
ATM_PREVIEW_LIMIT = 5
REQUIRED_COLUMNS = ["QUOTE_DATE", "EXPIRE_DATE", "STRIKE", "C_BID", "C_ASK", "C_DELTA"]


def parse_iso_date(value: str) -> date:
    return date.fromisoformat(value)


def format_date(value: object) -> str:
    return value.isoformat() if isinstance(value, date) else "brak"


def format_float(value: object) -> str:
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return "brak"


def iter_parquet_files(data_dir: Path) -> list[Path]:
    return sorted(data_dir.glob("spx_eod_*.parquet"))


def get_file_quote_date_bounds(file_path: Path) -> tuple[date | None, date | None]:
    table = pq.read_table(file_path, columns=["QUOTE_DATE"])
    quote_dates = table.column("QUOTE_DATE")
    return pc.min(quote_dates).as_py(), pc.max(quote_dates).as_py()


def detect_global_date_range(files: list[Path]) -> tuple[date, date]:
    overall_min: date | None = None
    overall_max: date | None = None

    for file_path in files:
        file_min, file_max = get_file_quote_date_bounds(file_path)
        if file_min is None or file_max is None:
            continue
        overall_min = file_min if overall_min is None else min(overall_min, file_min)
        overall_max = file_max if overall_max is None else max(overall_max, file_max)

    if overall_min is None or overall_max is None:
        raise ValueError("Nie udalo sie wykryc zakresu dat w plikach Parquet.")

    return overall_min, overall_max


def select_files_for_range(files: list[Path], start_date: date, end_date: date) -> list[Path]:
    selected_files: list[Path] = []
    for file_path in files:
        file_min, file_max = get_file_quote_date_bounds(file_path)
        if file_min is None or file_max is None:
            continue
        if file_max < start_date or file_min > end_date:
            continue
        selected_files.append(file_path)
    return selected_files


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analiza danych SPX z plikow Parquet.")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR, help="Katalog z plikami Parquet.")
    parser.add_argument(
        "--start-date",
        type=parse_iso_date,
        help="Data poczatkowa w formacie YYYY-MM-DD. Domyslnie poczatek calego zbioru.",
    )
    parser.add_argument(
        "--end-date",
        type=parse_iso_date,
        help="Data koncowa w formacie YYYY-MM-DD. Domyslnie koniec calego zbioru.",
    )
    return parser


def cast_numeric_column(table: object, column_name: str):
    trimmed = pc.utf8_trim_whitespace(table.column(column_name).chunk(0))
    empty_mask = pc.equal(trimmed, "")
    cleaned = pc.replace_with_mask(trimmed, empty_mask, pa.scalar(None, type=pa.string()))
    return pc.cast(cleaned, "float64")


def main() -> None:
    args = build_parser().parse_args()
    data_dir = args.data_dir.resolve()

    if not data_dir.exists():
        raise FileNotFoundError(f"Nie znaleziono katalogu: {data_dir}")

    files = iter_parquet_files(data_dir)
    if not files:
        raise FileNotFoundError(f"Nie znaleziono plikow Parquet w: {data_dir}")

    global_start, global_end = detect_global_date_range(files)
    start_date = args.start_date or global_start
    end_date = args.end_date or global_end

    if start_date > end_date:
        raise ValueError("Data poczatkowa nie moze byc pozniejsza niz data koncowa.")

    selected_files = select_files_for_range(files, start_date, end_date)
    if not selected_files:
        raise ValueError("Brak plikow Parquet pokrywajacych zadany zakres dat.")

    dataset = ds.dataset([str(path) for path in selected_files], format="parquet")
    date_filter = (ds.field("QUOTE_DATE") >= ds.scalar(start_date)) & (ds.field("QUOTE_DATE") <= ds.scalar(end_date))
    table = dataset.to_table(columns=REQUIRED_COLUMNS, filter=date_filter).combine_chunks()

    if table.num_rows == 0:
        raise ValueError("Brak rekordow w zadanym zakresie dat.")

    row_count = table.num_rows
    min_quote_date = pc.min(table.column("QUOTE_DATE")).as_py()
    max_quote_date = pc.max(table.column("QUOTE_DATE")).as_py()
    min_strike = pc.min(table.column("STRIKE")).as_py()
    max_strike = pc.max(table.column("STRIKE")).as_py()
    bid_values = cast_numeric_column(table, "C_BID")
    ask_values = cast_numeric_column(table, "C_ASK")
    delta_values = cast_numeric_column(table, "C_DELTA")

    atm_mask = pc.and_kleene(
        pc.greater_equal(delta_values, ATM_DELTA_MIN),
        pc.less_equal(delta_values, ATM_DELTA_MAX),
    )
    atm_table = pa.table(
        {
            "QUOTE_DATE": table.column("QUOTE_DATE"),
            "EXPIRE_DATE": table.column("EXPIRE_DATE"),
            "STRIKE": table.column("STRIKE"),
            "C_BID": bid_values,
            "C_ASK": ask_values,
            "C_DELTA": delta_values,
        }
    ).filter(atm_mask)
    atm_calls = atm_table.slice(0, ATM_PREVIEW_LIMIT).to_pylist()

    print("--- Statystyki zestawu danych ---")
    print(f"Zakres zapytania: {start_date.isoformat()} do {end_date.isoformat()}")
    print(f"Przetworzone pliki: {len(selected_files)}")
    print(f"Liczba rekordow: {row_count}")
    print(f"Zakres dat w wynikach: {format_date(min_quote_date)} do {format_date(max_quote_date)}")
    print(f"Dostepne Striki: od {format_float(min_strike)} do {format_float(max_strike)}")
    print("\nNaglowki:")
    print(table.schema.names)

    print("\nPrzykladowe opcje ATM (At-the-money):")
    if not atm_calls:
        print("Brak opcji CALL z delta w zakresie 0.48-0.52.")
        return

    print("QuoteDate   Expiration  Strike   Bid    Ask    Delta  DTE")
    for call in atm_calls:
        quote_date = call["QUOTE_DATE"]
        expiration = call["EXPIRE_DATE"]
        dte = (expiration - quote_date).days if isinstance(quote_date, date) and isinstance(expiration, date) else "brak"
        print(
            f"{format_date(quote_date):<11} "
            f"{format_date(expiration):<11} "
            f"{float(call['STRIKE']):>7.2f} "
            f"{float(call['C_BID']):>6.2f} "
            f"{float(call['C_ASK']):>6.2f} "
            f"{float(call['C_DELTA']):>6.3f} "
            f"{dte:>4}"
        )


if __name__ == "__main__":
    main()
