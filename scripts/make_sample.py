"""Build the versioned sample dataset from the full raw history.

The raw file (243 MB) cannot live in the repository, so `data/sample/` carries a
small slice that is committed. That slice is not a random subset: it is chosen so
that **every edge case the pipeline has to handle is present in it**. A sample
that only contains well-behaved rows would let the test suite pass while the
cleaning and feature code silently breaks on the real data.

Tickers were picked to cover:

    JNJ, KO       long history, defensive, lowest realised volatility
    CAT, XOM      cyclical industrial and energy
    MSFT          large-cap technology, steady
    AAPL, NVDA    technology with stock splits inside the window
    TSLA          very high volatility, multiple splits
    NEE           carries the empty-price rows dated 2026-08-28
    PLTR          starts mid-window (2020) and is the most volatile name
    EA, CICC      6 and 137 rows -- must be dropped by the MIN_HISTORY filter

Usage
-----
    python scripts/make_sample.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "master_stock_data.csv"
OUT = ROOT / "data" / "sample" / "sample_stock_data.csv"

START = "2015-01-01"

TICKERS = [
    "JNJ", "KO",      # low volatility, long history
    "CAT", "XOM",     # cyclical
    "MSFT",           # steady large-cap tech
    "AAPL", "NVDA",   # tech with splits in window
    "TSLA",           # very high volatility, splits
    "NEE",            # contains the empty-price rows
    "PLTR",           # partial history, highest volatility
    "EA", "CICC",     # too short -- exercise the cleaning filter
]


def build_sample() -> pd.DataFrame:
    df = pd.read_csv(RAW, parse_dates=["Date"])
    mask = df["Ticker"].isin(TICKERS) & (df["Date"] >= START)
    return df.loc[mask].sort_values(["Ticker", "Date"]).reset_index(drop=True)


def audit(df: pd.DataFrame) -> None:
    """Report the edge cases that actually landed in the sample.

    Asserting what we intended is not enough -- this prints what is really there.
    """
    print(f"rows      : {len(df):,}")
    print(f"tickers   : {df['Ticker'].nunique()}")
    print(f"period    : {df['Date'].min().date()} -> {df['Date'].max().date()}")
    print()

    print("rows per ticker")
    counts = df.groupby("Ticker").agg(
        n=("Date", "size"),
        start=("Date", "min"),
        end=("Date", "max"),
    )
    for ticker, row in counts.sort_values("n").iterrows():
        print(f"  {ticker:<6} {row['n']:>6,}   {row['start'].date()} -> {row['end'].date()}")
    print()

    print("edge cases captured")
    empty = df["Close"].isna().sum()
    splits = (df["Stock Splits"] != 0).sum()
    divs = (df["Dividends"] != 0).sum()
    zero_vol = (df["Volume"] == 0).sum()
    short = (counts["n"] < 756).sum()
    print(f"  empty-price rows        : {empty}")
    print(f"  stock splits            : {splits}")
    print(f"  dividend payments       : {divs}")
    print(f"  zero-volume rows        : {zero_vol}")
    print(f"  tickers under 756 rows  : {short}")

    if splits:
        print()
        print("  split events:")
        for _, r in df.loc[df["Stock Splits"] != 0, ["Date", "Ticker", "Stock Splits"]].iterrows():
            print(f"    {r['Date'].date()}  {r['Ticker']:<6} {r['Stock Splits']:g}:1")


def main() -> None:
    if not RAW.exists():
        raise SystemExit(f"raw dataset not found at {RAW}")

    sample = build_sample()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(OUT, index=False)

    audit(sample)
    print()
    print(f"written   : {OUT.relative_to(ROOT)}  ({OUT.stat().st_size / 1024**2:.2f} MB)")


if __name__ == "__main__":
    main()
