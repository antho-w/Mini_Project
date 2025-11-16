import argparse
import os
import pandas as pd

# preprocess_data.py
# Usage: python preprocess_data.py input.csv [--outdir monthly_csv] [--chunksize 100000]

def split_monthly(input_csv, outdir="monthly_csv", chunksize=100000):
    os.makedirs(outdir, exist_ok=True)
    written = set()  # keep track of which month files already have headers written

    # parser that handles the provided timestamp format (with trailing Z and nanoseconds)
    date_parser = lambda x: pd.to_datetime(x, utc=True, errors="coerce")

    for chunk in pd.read_csv(input_csv, chunksize=chunksize, parse_dates=["ts_event"], date_parser=date_parser):
        # ensure ts_event parsed; drop rows that failed to parse
        if chunk["ts_event"].isnull().any():
            chunk = chunk.dropna(subset=["ts_event"])
        # create month key like "2013-04"
        chunk["year_month"] = chunk["ts_event"].dt.strftime("%Y-%m")
        for month, grp in chunk.groupby("year_month"):
            outpath = os.path.join(outdir, f"{month}.csv")
            header = month not in written
            # drop helper column before writing
            to_write = grp.drop(columns=["year_month"])
            to_write.to_csv(outpath, mode="a", header=header, index=False)
            written.add(month)

def split_yearly(input_csv, outdir="monthly_csv", chunksize=100000):
    os.makedirs(outdir, exist_ok=True)
    written = set()  # keep track of which month files already have headers written

    # parser that handles the provided timestamp format (with trailing Z and nanoseconds)
    date_parser = lambda x: pd.to_datetime(x, utc=True, errors="coerce")

    for chunk in pd.read_csv(input_csv, chunksize=chunksize, parse_dates=["ts_event"], date_parser=date_parser):
        # ensure ts_event parsed; drop rows that failed to parse
        if chunk["ts_event"].isnull().any():
            chunk = chunk.dropna(subset=["ts_event"])
        # create year key like "2013"
        chunk["year"] = chunk["ts_event"].dt.strftime("%Y")
        for year, grp in chunk.groupby("year"):
            outpath = os.path.join(outdir, f"{year}.csv")
            header = year not in written
            # drop helper column before writing
            to_write = grp.drop(columns=["year"])
            to_write.to_csv(outpath, mode="a", header=header, index=False)
            written.add(year)

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Split CSV into monthly files based on ts_event column.")
    p.add_argument("input_csv", help="Path to input CSV file")
    p.add_argument("--outdir", default="monthly_csv", help="Directory to write monthly CSVs")
    p.add_argument("--chunksize", type=int, default=100000, help="Rows per chunk (for large files)")
    p.add_argument("--period", type=str, help="Must be 'yearly' or 'monthly'")
    args = p.parse_args()

    if args.period == "monthly":
        split_monthly(args.input_csv, args.outdir, args.chunksize)
    elif args.period == "yearly":
        split_yearly(args.input_csv, args.outdir, args.chunksize)