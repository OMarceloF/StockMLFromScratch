"""Primitive transformations of the price series.

Everything downstream -- features, targets, diagnostics -- is built on log
returns, so they live here rather than being re-derived in each notebook.

Two rules hold for every function in this module:

1. **Group by ticker, always.** The tickers do not share a trading calendar
   (AAPL has 2,932 days in 2015-2026 where CAT has 2,931), and the frame stores
   them stacked. Any window or lag that ignores the group boundary reads the end
   of one company's history as the start of the next.
2. **Look backwards only.** A value dated `t` uses data up to and including `t`.
   Forward-looking quantities are targets, and live in `targets.py`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import config


def log_returns(
    df: pd.DataFrame,
    *,
    price_col: str = "adj_close",
    group_col: str = "ticker",
) -> pd.Series:
    """Daily log returns, computed within each ticker.

    ``adj_close`` is the default for a reason: it is adjusted for both splits
    and dividends. Using ``close`` instead would turn every dividend payment
    into a fake overnight drop, because the price falls by the dividend on the
    ex-date while the shareholder is made whole in cash. Over 1.8M rows and
    ~26k dividend events, that is a systematic negative bias in the returns.

    Log returns rather than simple returns: they are additive across time
    (summing daily log returns gives the period log return, which is not true
    of simple returns) and roughly symmetric around zero, which matters for a
    model that assumes symmetric errors.

    Returns
    -------
    Series
        Aligned to `df.index`. The first observation of each ticker is NaN --
        there is no prior price to difference against.
    """
    log_price = np.log(df[price_col])
    return log_price.groupby(df[group_col], observed=True).diff()


def realized_volatility(
    returns: pd.Series,
    groups: pd.Series,
    window: int,
    *,
    annualize: bool = True,
    min_periods: int | None = None,
) -> pd.Series:
    """Rolling standard deviation of returns, within each ticker.

    Parameters
    ----------
    returns
        Log returns, as produced by `log_returns`.
    groups
        The ticker column, aligned to `returns`.
    window
        Number of trading days in the window.
    annualize
        Multiply by sqrt(252) so the number reads as an annual percentage,
        the convention every volatility figure in finance uses.
    min_periods
        Observations required before emitting a value. Defaults to `window`,
        i.e. no partial windows: a "21-day volatility" computed from 4 points
        is not a 21-day volatility, and silently emitting one would put
        wildly noisy values at the start of every ticker's history.

    Returns
    -------
    Series
        Aligned to `returns.index`.
    """
    if min_periods is None:
        min_periods = window

    vol = (
        returns.groupby(groups, observed=True)
        .rolling(window, min_periods=min_periods)
        .std()
        .droplevel(0)
        .reindex(returns.index)
    )

    if annualize:
        vol = vol * np.sqrt(config.TRADING_DAYS_PER_YEAR)
    return vol
