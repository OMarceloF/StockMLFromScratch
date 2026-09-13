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

    Log returns rather than simple returns, for one reason that holds and one
    that is often claimed and does not.

    **Additivity across time — the real reason.** Daily log returns sum to the
    period log return; simple returns do not. Over AAPL's 9,233 days here, the
    daily log returns sum to 7.1046, exactly `log(final / initial)`; the simple
    returns sum to 10.3999, a number with no interpretation. This matters
    directly: the 21-day targets in `targets.py` are sums of daily log returns,
    which is only meaningful because of this property.

    **Symmetry — measured, and the claim does not survive.** Textbooks often
    add that log returns are "more symmetric". On this dataset they are not:
    pooled skewness is +0.54 for simple returns and -0.52 for log returns, and
    per ticker the log version is closer to symmetric in only 108 of 231 cases
    (median |skew| 0.314 against 0.316 -- a coin flip). The transform moves the
    skew from positive to negative rather than removing it, because taking logs
    compresses large gains and stretches large losses. Log returns are the right
    choice here on additivity alone.

    Returns
    -------
    Series
        Aligned to `df.index`. The first observation of each ticker is NaN --
        there is no prior price to difference against.
    """
    log_price = np.log(df[price_col])
    return log_price.groupby(df[group_col], observed=True).diff()


def momentum(
    returns: pd.Series,
    groups: pd.Series,
    window: int,
    *,
    min_periods: int | None = None,
) -> pd.Series:
    """Cumulative log return over the trailing `window` days, within each ticker.

    This is a rolling **sum**, and that is the whole payoff of working in log
    space: because daily log returns add up to the period return, the
    cumulative return over any window is just their sum. With simple returns
    the same quantity would need a compounding product, which is slower and
    loses precision over long windows.

    ``window=1`` returns the daily return unchanged, by construction.

    No lag is applied. The value dated `t` covers `t-window+1 .. t` inclusive,
    all of which is known at the close of `t`; the targets start at `t+1`.

    Parameters
    ----------
    returns
        Daily log returns, from `log_returns`.
    groups
        The ticker column, aligned to `returns`.
    window
        Number of trading days to accumulate.
    min_periods
        Observations required before emitting a value. Defaults to `window`:
        a "126-day momentum" computed from 12 days is not a 126-day momentum,
        and emitting one would put a differently-scaled feature at the start of
        every ticker's history.

    Returns
    -------
    Series
        Aligned to `returns.index`.
    """
    if min_periods is None:
        min_periods = window

    return (
        returns.groupby(groups, observed=True)
        .rolling(window, min_periods=min_periods)
        .sum()
        .droplevel(0)
        .reindex(returns.index)
    )


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
