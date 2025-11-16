"""
Simple preprocessing script to split a large OHLCV CSV into per-year CSV files.

Usage:
    python preprocess_data.py <input_csv> --outdir <output_dir>

If outdir does not exist it will be created.
"""
import argparse
import os
from pathlib import Path
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def find_date_column(df: pd.DataFrame) -> str:
    """Try to heuristically find a date column in the dataframe."""
    candidates = [c for c in df.columns if 'date' in c.lower() or 'time' in c.lower()]
    if candidates:
        return candidates[0]
    # fallback: if first column looks like a date
    first = df.columns[0]
    try:
        pd.to_datetime(df[first])
        return first
    except Exception:
        pass
    raise ValueError("Could not find a date/time column in the input CSV; please provide one with name containing 'date' or 'time'.")


def parse_symbol(sym: str):
    # expected format contains expiry as YYMMDD before a 'C' or 'P',
    # e.g. 'AAPL231217C00125000' or similar. We'll search for the pattern
    # of 6 digits followed by C or P.
    import re
    if not isinstance(sym, str):
        return (pd.NaT, None, None)
    m = re.search(r"(\d{6})([CP])(\d+)", sym)
    if not m:
        return (pd.NaT, None, None)
    yymmdd, cp, strike_raw = m.groups()
    # parse expiry YYMMDD -> YYYY-MM-DD
    yy = int(yymmdd[:2])
    year = 2000 + yy if yy < 70 else 1900 + yy
    month = int(yymmdd[2:4])
    day = int(yymmdd[4:6])
    try:
        expiry = pd.Timestamp(year=year, month=month, day=day)
    except Exception:
        expiry = pd.NaT
    option_type = 'calls' if cp == 'C' else 'puts'
    # strike is digits after C/P; contract convention often uses 3 decimals, divide by 1000
    try:
        strike = int(strike_raw) / 1000.0
    except Exception:
        strike = None
    return (expiry, option_type, strike)

def split_yearly(input_csv: Path, outdir: Path, date_col: str = None, date_format: str = None, rfr_path: Path = None):
    logger.info("Reading input CSV: %s", input_csv)
    df = pd.read_csv(input_csv)

    # If a 'symbol' column exists, try to extract expiry, option type and strike
    if 'symbol' in df.columns:

        parsed = df['symbol'].apply(parse_symbol)
        df['expiry'] = parsed.apply(lambda x: x[0])
        df['optionType'] = parsed.apply(lambda x: x[1])
        df['strike'] = parsed.apply(lambda x: x[2])

    if date_col is None:
        date_col = find_date_column(df)
        logger.info("Detected date column: %s", date_col)

    # Parse dates
    df[date_col] = pd.to_datetime(df[date_col], format=date_format, errors='coerce')
    if df[date_col].isna().all():
        raise ValueError("Failed to parse any dates from the selected date column. Try specifying --date-col or --date-format.")

    df = df.dropna(subset=[date_col])
    # create observation date (date only) for joining external data like EFFR
    df['obs_date'] = df[date_col].dt.date

    # If a risk-free-rate CSV path was provided (or default exists), load and merge by obs_date
    rfr_path_resolved = None
    if rfr_path:
        rfr_path_resolved = Path(rfr_path)
    else:
        # default relative location inside the project
        default_rfr = Path(__file__).resolve().parents[0] / 'raw_data' / 'EFFR.csv'
        if default_rfr.exists():
            rfr_path_resolved = default_rfr

    if rfr_path_resolved and rfr_path_resolved.exists():
        logger.info("Loading risk-free rate file: %s", rfr_path_resolved)
        rfr_df = pd.read_csv(rfr_path_resolved)
        # detect date column in rfr file
        try:
            rfr_date_col = find_date_column(rfr_df)
        except Exception:
            # fallback try common name
            rfr_date_col = 'Date' if 'Date' in rfr_df.columns else rfr_df.columns[0]
        rfr_df[rfr_date_col] = pd.to_datetime(rfr_df[rfr_date_col], errors='coerce')
        rfr_df = rfr_df.dropna(subset=[rfr_date_col])
        rfr_df['obs_date'] = rfr_df[rfr_date_col].dt.date
        # pick a numeric column for the rate (exclude date)
        rate_cols = [c for c in rfr_df.columns if c != rfr_date_col and pd.api.types.is_numeric_dtype(rfr_df[c])]
        if not rate_cols:
            logger.warning("No numeric rate column found in %s, skipping risk-free merge", rfr_path_resolved)
        else:
            rate_col = rate_cols[0]
            rfr_small = rfr_df[['obs_date', rate_col]].drop_duplicates(subset=['obs_date'])
            rfr_small = rfr_small.rename(columns={rate_col: 'risk_free_rate'})
            # merge onto main df
            before_matches = int(df['obs_date'].notna().sum())
            df = df.merge(rfr_small, on='obs_date', how='left')
            matches = int(df['risk_free_rate'].notna().sum())
            logger.info("Merged risk-free rates: matched %d of %d observation dates", matches, before_matches)
            # Forward-fill risk-free rate (last observation carried forward)
            # sort by original datetime column so ffill respects chronological order
            try:
                df = df.sort_values(by=date_col)
            except Exception:
                # fallback: sort by obs_date
                df = df.sort_values(by='obs_date')
            df['risk_free_rate'] = df['risk_free_rate'].ffill()
            filled_matches = int(df['risk_free_rate'].notna().sum())
            logger.info("After LOCF fill, risk-free rate non-NA count: %d", filled_matches)
    else:
        logger.info("No risk-free rate file provided or found at default location; skipping merge")
    df['year'] = df[date_col].dt.year

    outdir.mkdir(parents=True, exist_ok=True)

    grouped = df.groupby('year')
    for year, group in grouped:
        out_path = outdir / f"{year}.csv"
        # drop helper column
        group = group.drop(columns=['year'])
        group.to_csv(out_path, index=False)
        logger.info("Wrote %d rows to %s", len(group), out_path)

    logger.info("Finished splitting into %d files", grouped.ngroups)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Split OHLCV CSV into per-year CSV files')
    parser.add_argument('input_csv', help='Path to input CSV file')
    parser.add_argument('--outdir', default='full_yearly_data', help='Directory to write yearly CSV files')
    parser.add_argument('--date-col', dest='date_col', help="Name of the date column (optional, auto-detected otherwise)")
    parser.add_argument('--date-format', dest='date_format', help="Optional date format to pass to pandas.to_datetime")
    parser.add_argument('--rfr', dest='rfr_path', help="Optional path to risk-free rate CSV to merge (EFFR.csv)")

    args = parser.parse_args()
    input_csv = Path(args.input_csv)
    outdir = Path(args.outdir)

    if not input_csv.exists():
        logger.error("Input file does not exist: %s", input_csv)
        raise SystemExit(1)

    split_yearly(input_csv, outdir, date_col=args.date_col, date_format=args.date_format, rfr_path=args.rfr_path)
