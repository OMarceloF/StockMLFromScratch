"""Assemble the model-ready dataset: X, y, and the rows that survive both.

This module makes two commitments the rest of the project relies on.

**The feature set is fixed here, and only here.** Items 2.2 through 2.5 built
more functions than the model uses -- `volatility_ratio`, `rsi`,
`position_in_range` exist, are tested, and are deliberately not selected. The
list below is the result of measurement, not of everything that was written:
each candidate had to add at least +0.0005 to the walk-forward R^2 and win a
majority of the 22 annual folds.

**Warm-up and tail rows are removed explicitly.** Each ticker loses its first
`WARMUP` rows, where the longest rolling window is not yet full, and its last
`HORIZON` rows, whose future has not happened. Both boundaries are already NaN,
so dropping them is not strictly necessary -- but counting them makes the loss
auditable rather than incidental.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src import config
from src.features.targets import TARGET_COLUMNS, build_targets
from src.features.technical import (
    downside_volatility,
    log_returns,
    momentum,
    parkinson_volatility,
    price_vs_moving_average,
    realized_volatility,
    safe_log,
    volume_zscore,
)

#: The selected features, grouped by what they describe.
#:
#: Ablation on the real dataset, by block (walk-forward R^2):
#:
#:     log_park_21 alone .................. 0.5148
#:     + the three close-to-close vols .... 0.5436  (+0.0288)
#:     + downside volatility .............. 0.5475  (+0.0039)
#:     + the five momentum features ....... 0.5503  (+0.0028)
#:     + volume z-score ................... 0.5503  (+0.0000)
#:     + the two moving-average distances . 0.5511  (+0.0008)
#:
#: The model is, honestly, four volatility features and a tail of small
#: contributions. That is worth stating rather than hiding behind a long list.
VOLATILITY_FEATURES = ["log_vol_5", "log_vol_21", "log_vol_63", "log_dvol_21", "log_park_21"]
MOMENTUM_FEATURES = ["ret_1", "ret_5", "ret_21", "ret_63", "ret_126"]
VOLUME_FEATURES = ["volume_z_63"]
PRICE_FEATURES = ["px_vs_sma_21", "px_vs_sma_63"]

FEATURE_COLUMNS = (
    VOLATILITY_FEATURES + MOMENTUM_FEATURES + VOLUME_FEATURES + PRICE_FEATURES
)

#: Features for Experiment 0 only. Deliberately raw prices on their natural
#: scale -- the opposite of every rule the real feature set follows. Reproducing
#: the mistake the project dismantles requires making it faithfully.
NAIVE_PRICE_COLUMNS = ["close_lag_1", "close_lag_2", "close_lag_5"]


@dataclass
class DatasetReport:
    """What the assembly kept and what it dropped, and why."""

    rows_in: int
    rows_out: int
    tickers_in: int
    tickers_out: int
    dropped_warmup: int
    dropped_tail: int
    dropped_other: int

    def __str__(self) -> str:
        pct = 100 * (self.rows_in - self.rows_out) / self.rows_in if self.rows_in else 0.0
        return "\n".join([
            "Dataset report",
            f"  in   : {self.rows_in:>10,} rows   {self.tickers_in:>3} tickers",
            f"  {'warm-up (incomplete windows)':<34}{self.dropped_warmup:>10,}",
            f"  {'tail (target not yet observed)':<34}{self.dropped_tail:>10,}",
            f"  {'other missing values':<34}{self.dropped_other:>10,}",
            f"  out  : {self.rows_out:>10,} rows   {self.tickers_out:>3} tickers"
            f"   ({pct:.2f}% dropped)",
        ])


def build_features(df: pd.DataFrame, *, returns: pd.Series | None = None) -> pd.DataFrame:
    """Compute every selected feature for a canonical price frame.

    Returns
    -------
    DataFrame
        Columns `FEATURE_COLUMNS`, aligned to `df.index`. Leading rows of each
        ticker are NaN while the rolling windows fill.
    """
    groups = df["ticker"]
    if returns is None:
        returns = log_returns(df)

    features = {}

    for window in config.VOLATILITY_WINDOWS:
        features[f"log_vol_{window}"] = safe_log(
            realized_volatility(returns, groups, window)
        )
    features["log_dvol_21"] = safe_log(downside_volatility(returns, groups, 21))
    features["log_park_21"] = safe_log(
        parkinson_volatility(df["high"], df["low"], groups, config.PARKINSON_WINDOW)
    )

    for window in config.MOMENTUM_WINDOWS:
        features[f"ret_{window}"] = momentum(returns, groups, window)

    features["volume_z_63"] = volume_zscore(
        df["volume"], groups, config.VOLUME_ZSCORE_WINDOW
    )

    for window in config.SMA_WINDOWS:
        features[f"px_vs_sma_{window}"] = price_vs_moving_average(
            df["adj_close"], groups, window
        )

    return pd.DataFrame(features, index=df.index)[FEATURE_COLUMNS]


def build_naive_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """Lagged raw closing prices, for Experiment 0 only.

    Every other feature in this project is scale-free by design. These are not,
    and that is the point: a model fed yesterday's price to predict tomorrow's
    price will report an R^2 of roughly 0.9996 and a coefficient near 1.0 on
    `close_lag_1`, having learned nothing whatsoever.
    """
    groups = df["ticker"]
    return pd.DataFrame(
        {
            f"close_lag_{lag}": df["close"].groupby(groups, observed=True).shift(lag)
            for lag in (1, 2, 5)
        },
        index=df.index,
    )[NAIVE_PRICE_COLUMNS]


def build_dataset(
    df: pd.DataFrame,
    *,
    horizon: int = config.HORIZON,
    dropna: bool = True,
) -> tuple[pd.DataFrame, DatasetReport]:
    """Assemble features, targets and identifiers into one model-ready frame.

    Parameters
    ----------
    df
        Canonical price frame, cleaned (see `src.data.cache.get_prices`).
    horizon
        Forecast horizon, passed through to the targets.
    dropna
        Drop rows missing any feature or the main volatility target. Turn this
        off to inspect the warm-up and tail rows directly.

    Returns
    -------
    (DataFrame, DatasetReport)
        Columns: `date`, `ticker`, `FEATURE_COLUMNS`, `TARGET_COLUMNS`, plus a
        fresh RangeIndex -- and an account of the rows that did not survive.
    """
    returns = log_returns(df)
    features = build_features(df, returns=returns)
    targets = build_targets(df, horizon=horizon, returns=returns)

    dataset = pd.concat([df[["date", "ticker"]], features, targets], axis=1)

    groups = df["ticker"]
    position = groups.groupby(groups, observed=True).cumcount()
    size = groups.map(groups.value_counts())
    is_warmup = position < config.WARMUP
    is_tail = position >= size - horizon

    report = DatasetReport(
        rows_in=len(dataset),
        rows_out=len(dataset),
        tickers_in=int(groups.nunique()),
        tickers_out=int(groups.nunique()),
        dropped_warmup=int(is_warmup.sum()),
        dropped_tail=int(is_tail.sum()),
        dropped_other=0,
    )

    if dropna:
        required = FEATURE_COLUMNS + ["y_log_vol_fwd"]
        keep = dataset[required].notna().all(axis=1)
        report.dropped_other = int((~keep & ~is_warmup & ~is_tail).sum())
        dataset = dataset[keep].reset_index(drop=True)
        dataset["ticker"] = dataset["ticker"].cat.remove_unused_categories()
        report.rows_out = len(dataset)
        report.tickers_out = int(dataset["ticker"].nunique())

    return dataset, report
