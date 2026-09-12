"""Remove unusable rows and tickers, and account for every removal.

The guiding rule of this module is that **nothing is dropped silently**. Every
filter reports how many rows and tickers it removed and why, and `clean_prices`
returns that account alongside the data. Cleaning that leaves no trace is how a
pipeline quietly discards a third of its observations and still looks healthy.

Each decision below was taken after measuring the real dataset (1,800,311 rows,
233 tickers, 1990 onward); the counts quoted in the comments are what that scan
found.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from src import config

PRICE_COLUMNS = ["open", "high", "low", "close", "adj_close"]


@dataclass(frozen=True)
class CleaningStep:
    """One filter's contribution to the cleaning report."""

    name: str
    reason: str
    rows_removed: int
    tickers_removed: int


@dataclass
class CleaningReport:
    """Auditable account of everything `clean_prices` discarded."""

    rows_in: int
    tickers_in: int
    rows_out: int = 0
    tickers_out: int = 0
    steps: list[CleaningStep] = field(default_factory=list)
    removed_tickers: list[str] = field(default_factory=list)

    @property
    def rows_removed(self) -> int:
        return self.rows_in - self.rows_out

    @property
    def tickers_removed(self) -> int:
        return self.tickers_in - self.tickers_out

    def __str__(self) -> str:
        pct = 100 * self.rows_removed / self.rows_in if self.rows_in else 0.0
        lines = [
            "Cleaning report",
            f"  in   : {self.rows_in:>10,} rows   {self.tickers_in:>3} tickers",
            "",
            f"  {'step':<24}{'reason':<42}{'rows':>9}{'tickers':>9}",
            f"  {'-' * 84}",
        ]
        for step in self.steps:
            lines.append(
                f"  {step.name:<24}{step.reason:<42}"
                f"{step.rows_removed:>9,}{step.tickers_removed:>9}"
            )
        lines += [
            f"  {'-' * 84}",
            f"  out  : {self.rows_out:>10,} rows   {self.tickers_out:>3} tickers"
            f"   ({self.rows_removed:,} rows removed, {pct:.3f}%)",
        ]
        if self.removed_tickers:
            lines.append(f"  dropped tickers: {', '.join(self.removed_tickers)}")
        return "\n".join(lines)


def _drop(
    df: pd.DataFrame,
    keep: pd.Series,
    name: str,
    reason: str,
    report: CleaningReport,
) -> pd.DataFrame:
    """Apply a boolean keep-mask and record what it cost."""
    before_rows = len(df)
    before_tickers = df["ticker"].nunique()
    df = df[keep]
    report.steps.append(
        CleaningStep(
            name=name,
            reason=reason,
            rows_removed=before_rows - len(df),
            tickers_removed=before_tickers - df["ticker"].nunique(),
        )
    )
    return df


def clean_prices(
    df: pd.DataFrame,
    *,
    min_history_days: int = config.MIN_HISTORY_DAYS,
    drop_zero_volume: bool = True,
) -> tuple[pd.DataFrame, CleaningReport]:
    """Filter the canonical price frame and report every removal.

    Parameters
    ----------
    df
        Canonical frame from `src.data.loader.load_prices`.
    min_history_days
        Tickers with fewer surviving rows than this are dropped entirely.
    drop_zero_volume
        Whether to discard days on which nothing traded. See the note in the
        step below for the evidence behind the default.

    Returns
    -------
    (DataFrame, CleaningReport)
        The cleaned frame (same schema, fresh RangeIndex) and the account of
        what was removed.
    """
    missing = [c for c in ("date", "ticker", "volume", *PRICE_COLUMNS) if c not in df.columns]
    if missing:
        raise ValueError(f"input is not a canonical price frame; missing columns: {missing}")

    report = CleaningReport(rows_in=len(df), tickers_in=df["ticker"].nunique())
    tickers_before = set(df["ticker"].unique())

    # -- 1. duplicates ------------------------------------------------------
    # None exist today, but a duplicated (ticker, date) would corrupt every
    # shift() and rolling() downstream without raising anything. Keeping the
    # first occurrence is arbitrary; what matters is that the count is reported
    # rather than the rows being silently collapsed.
    df = _drop(
        df,
        ~df.duplicated(["ticker", "date"], keep="first"),
        "duplicates",
        "repeated (ticker, date)",
        report,
    )

    # -- 2. missing prices --------------------------------------------------
    # 4 rows, all dated 2026-08-28 (BAX, EBAY, NEE, TMUS): dividend-only records
    # with no price and no volume.
    df = _drop(
        df,
        df[PRICE_COLUMNS].notna().all(axis=1),
        "missing prices",
        "NaN in open/high/low/close/adj_close",
        report,
    )

    # -- 3. non-positive prices --------------------------------------------
    # None exist (the minimum adj_close is 0.031), but log returns are the
    # foundation of every feature in Phase 2 and log(0) is -inf, which would
    # propagate through rolling windows as NaN rather than as an error.
    df = _drop(
        df,
        df[PRICE_COLUMNS].gt(0).all(axis=1),
        "non-positive prices",
        "price <= 0 would break log returns",
        report,
    )

    # -- 4. incoherent OHLC -------------------------------------------------
    # None exist. A violation means the source is corrupt for that row, and no
    # imputation could be trusted.
    coherent = (
        (df["low"] <= df["high"])
        & df["open"].between(df["low"], df["high"])
        & df["close"].between(df["low"], df["high"])
    )
    df = _drop(df, coherent, "incoherent OHLC", "low <= open/close <= high violated", report)

    # -- 5. zero-volume days ------------------------------------------------
    # 215 rows (0.012%) across 33 tickers, concentrated in the 1990s. These are
    # days on which nothing traded: 91.6% of them repeat the previous close
    # exactly, against 2.5% on ordinary days. Keeping them would inject an
    # artificial 0% return followed by the real move, and would make log(volume)
    # undefined in Phase 2.4.
    #
    # This is the one genuinely debatable filter here. Dropping the row means
    # the next return spans two calendar days, which is a more faithful
    # description of what happened than a fabricated flat day.
    if drop_zero_volume:
        df = _drop(
            df,
            df["volume"] > 0,
            "zero volume",
            "no trading; 91.6% carry a stale close",
            report,
        )

    # -- 6. short history ---------------------------------------------------
    # Deliberately last. Steps 1-5 can push a ticker below the threshold, so
    # counting history first would keep a ticker on the strength of rows that
    # are about to be discarded.
    counts = df.groupby("ticker", observed=True)["date"].transform("size")
    df = _drop(
        df,
        counts >= min_history_days,
        "short history",
        f"fewer than {min_history_days} surviving rows",
        report,
    )

    df = df.copy()
    df["ticker"] = df["ticker"].cat.remove_unused_categories()
    df = df.reset_index(drop=True)

    report.rows_out = len(df)
    report.tickers_out = df["ticker"].nunique()
    report.removed_tickers = sorted(tickers_before - set(df["ticker"].unique()))
    return df, report
