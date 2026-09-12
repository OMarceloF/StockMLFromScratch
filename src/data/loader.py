"""Read the raw price history into a canonical DataFrame.

This module does one job: turn a CSV on disk into a typed, renamed, filtered,
sorted DataFrame. It makes no judgement about which rows are usable -- dropping
bad rows and short-history tickers is `src.data.clean`'s responsibility, and
keeping the two separate means the cleaning report can state exactly what was
removed from a known starting point.

The canonical frame returned here is the contract every downstream module
depends on:

    date          datetime64[us]   trading day
    ticker        category         symbol
    open/high/low/close  float64   split-adjusted prices
    adj_close     float64          split- and dividend-adjusted price
    volume        int64            shares traded
    dividends     float64          cash dividend paid that day (0.0 if none)
    stock_splits  float64          split ratio that day (0.0 if none)

sorted by (ticker, date), with a fresh RangeIndex.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src import config

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

#: Raw header -> canonical name. Renaming to snake_case is not cosmetic: it
#: removes the spaces in "Adj Close" and "Stock Splits", which otherwise force
#: bracket access everywhere and make column names awkward to pass around as
#: identifiers.
COLUMN_MAP = {
    "Date": "date",
    "Ticker": "ticker",
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Adj Close": "adj_close",
    "Volume": "volume",
    "Dividends": "dividends",
    "Stock Splits": "stock_splits",
}

#: Canonical column order: identifiers, then OHLC in its conventional order,
#: then volume, then corporate actions.
COLUMNS = [
    "date", "ticker",
    "open", "high", "low", "close", "adj_close",
    "volume",
    "dividends", "stock_splits",
]

#: Explicit dtypes, keyed by *raw* header because read_csv applies them during
#: parsing. Every choice below was measured against the real file:
#:
#: ticker -> category
#:     233 distinct values repeated across 2.2M rows. Cuts that column from
#:     23.4 MB to 4.2 MB and makes groupby cheaper. The classic trap with
#:     categoricals is `groupby(observed=False)`, which emits a row for every
#:     unused level -- on a filtered frame that means 233 groups where 3 exist,
#:     230 of them NaN. pandas 3.0 defaults `observed=True`, and this loader
#:     drops unused levels after filtering, so both ends are covered.
#:
#: prices -> float64
#:     float32 would save ~60 MB and the values survive storage, but arithmetic
#:     in float32 introduces ~5e-7 error in log returns. This project validates
#:     its from-scratch models against scikit-learn at rtol=1e-6; carrying an
#:     error of the same magnitude through the features would make that
#:     comparison meaningless. Memory is not the binding constraint here.
#:
#: volume -> int64
#:     The maximum in this dataset is 9,230,856,000 -- past int32 (2.1e9) and
#:     past uint32 (4.3e9) as well. int64 is not a default, it is a requirement.
DTYPES = {
    "Ticker": "category",
    "Open": "float64",
    "High": "float64",
    "Low": "float64",
    "Close": "float64",
    "Adj Close": "float64",
    "Volume": "int64",
    "Dividends": "float64",
    "Stock Splits": "float64",
}


class SchemaError(ValueError):
    """Raised when the source file does not have the expected columns."""


def _validate_schema(df: pd.DataFrame, path: Path) -> None:
    """Fail immediately and legibly if the source file changed shape.

    Without this, a renamed or missing column surfaces as a KeyError three
    modules downstream, pointing at code that is not the problem.
    """
    found = set(df.columns)
    expected = set(COLUMN_MAP)
    if found != expected:
        missing = sorted(expected - found)
        extra = sorted(found - expected)
        parts = [f"unexpected schema in {path.name}"]
        if missing:
            parts.append(f"missing columns: {missing}")
        if extra:
            parts.append(f"unrecognised columns: {extra}")
        raise SchemaError("; ".join(parts))


def load_prices(
    path: Path | str = config.RAW_CSV,
    *,
    start_date: str | None = config.START_DATE,
) -> pd.DataFrame:
    """Load a price history CSV into the canonical frame.

    Parameters
    ----------
    path
        CSV to read. Defaults to the full raw dataset; pass
        ``config.SAMPLE_CSV`` for the small versioned sample.
    start_date
        Drop observations before this date (inclusive lower bound).
        ``None`` keeps the full history.

    Returns
    -------
    DataFrame
        Typed, renamed, filtered and sorted by ``(ticker, date)`` with a fresh
        RangeIndex.

    Raises
    ------
    FileNotFoundError
        If `path` does not exist, with a message that says how to get the file.
    SchemaError
        If the CSV's columns do not match the expected schema.
    """
    path = Path(path)
    if not path.is_file():
        hint = ""
        if path == config.RAW_CSV:
            hint = (
                "\nThe raw dataset is not tracked by git (243 MB). Download it from "
                "https://www.kaggle.com/datasets/parsalatifi/s-and-p-500-us-stock-market-data "
                f"and place it at {path}, or use config.SAMPLE_CSV instead."
            )
        raise FileNotFoundError(f"price data not found at {path}{hint}")

    df = pd.read_csv(path, dtype=DTYPES, parse_dates=["Date"])
    _validate_schema(df, path)
    df = df.rename(columns=COLUMN_MAP)

    if start_date is not None:
        df = df[df["date"] >= pd.Timestamp(start_date)]

    # Filtering leaves the categorical carrying every ticker that was ever in
    # the file, including those now fully removed. Dropping the dead levels
    # keeps `nunique`, `value_counts` and any explicit observed=False honest.
    df["ticker"] = df["ticker"].cat.remove_unused_categories()

    # Sorting is a precondition, not a nicety. Every feature in Phase 2 is built
    # from shift() and rolling(), which read neighbouring rows positionally --
    # on an unsorted frame they would silently mix dates, or read the tail of
    # one ticker as the head of the next. The raw file happens to arrive sorted;
    # relying on that would make the guarantee an accident instead of a fact.
    df = df.sort_values(["ticker", "date"], kind="stable").reset_index(drop=True)

    return df[COLUMNS]
