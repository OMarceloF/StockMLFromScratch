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


def safe_log(x: pd.Series) -> pd.Series:
    """Natural log, with non-positive inputs becoming NaN instead of -inf.

    Volatility is occasionally exactly zero. On this dataset `vol_5` is zero on
    161 rows across 35 tickers -- all of them before 2001, when US stocks still
    traded in eighths of a dollar and a $1.50 share simply could not move on a
    quiet day. ADI, for instance, closed at exactly $1.500000 for six straight
    sessions in January 1990, on real volume.

    The distinction matters more than the row count suggests. ``-inf`` is a
    float that arithmetic happily propagates: it survives into the design
    matrix and surfaces later as ``LinAlgError: SVD did not converge``, an
    error that says nothing about which column caused it. ``NaN`` is dropped by
    the pipeline's own missing-value handling, which is exactly the behaviour
    wanted for a row whose feature is genuinely undefined.
    """
    return np.log(x.where(x > 0))


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


def downside_volatility(
    returns: pd.Series,
    groups: pd.Series,
    window: int,
    *,
    annualize: bool = True,
    min_periods: int | None = None,
) -> pd.Series:
    """Volatility computed from the losing days only.

    Defined as the downside semi-deviation, ``sqrt(mean(min(r, 0)^2))``: up
    days enter the average as zeros rather than being dropped. Taking the plain
    standard deviation of the negative subset would be worse in two ways -- the
    sample size would vary from row to row, and a window with a single loss
    would report a volatility of zero.

    This feature exists to solve a specific measured problem. Future volatility
    is a **V** in past return: on this dataset, the bottom decile of 21-day
    momentum is followed by 41.9% annualised volatility and the top decile by
    34.2%, against 24.1% in the middle. Both tails raise volatility, and the
    losing tail raises it more. `realized_volatility` sees the symmetric part
    of that shape and is blind to the asymmetry; this sees the losing side
    specifically, which is what lets a linear model tilt the V.

    Returns
    -------
    Series
        Aligned to `returns.index`.
    """
    if min_periods is None:
        min_periods = window

    # clip(upper=0) is min(r, 0): losses keep their size, gains become zero,
    # and NaN stays NaN so it still counts against min_periods.
    squared_losses = returns.clip(upper=0) ** 2

    mean_squared = (
        squared_losses.groupby(groups, observed=True)
        .rolling(window, min_periods=min_periods)
        .mean()
        .droplevel(0)
        .reindex(returns.index)
    )

    vol = np.sqrt(mean_squared)
    if annualize:
        vol = vol * np.sqrt(config.TRADING_DAYS_PER_YEAR)
    return vol


def volume_zscore(
    volume: pd.Series,
    groups: pd.Series,
    window: int,
    *,
    min_periods: int | None = None,
) -> pd.Series:
    """How unusual today's turnover is, judged against this ticker's own recent past.

    Raw volume cannot enter a pooled model. On this dataset median daily volume
    runs from 149,600 shares (MTD) to 479 million (NVDA) -- a factor of 3,202 --
    so a coefficient fitted on one is meaningless for the other. Its skewness is
    21.1; in logs it is 0.09.

    The z-score is taken on **log** volume, not raw. Both forms predict the
    target about equally (correlation 0.058 against 0.056), but the raw version
    leaves 8,097 rows beyond five standard deviations against 461 for the log
    version -- a seventeen-fold difference in extreme values feeding a least
    squares fit that squares them.

    Rolling mean and standard deviation are computed per ticker, so the feature
    asks "is this heavy for *this* stock", which is scale-free by construction
    and comparable across the panel.

    Returns
    -------
    Series
        Aligned to `volume.index`. NaN where the window is incomplete, where
        volume is zero, or where turnover was constant across the whole window
        (a zero denominator).
    """
    if min_periods is None:
        min_periods = window

    log_volume = safe_log(volume.astype(float))
    rolling = log_volume.groupby(groups, observed=True).rolling(
        window, min_periods=min_periods
    )
    mean = rolling.mean().droplevel(0).reindex(log_volume.index)
    std = rolling.std().droplevel(0).reindex(log_volume.index)

    # A constant-volume window gives std 0; dividing would yield +/-inf, which
    # survives into the design matrix. NaN is dropped instead.
    return (log_volume - mean) / std.where(std > 0)


def volatility_ratio(short_vol: pd.Series, long_vol: pd.Series) -> pd.Series:
    """Log ratio of a short-window volatility to a long-window one.

    A regime indicator: positive means the last few days have been more
    turbulent than the recent baseline, negative means calmer. Because it is a
    ratio of two volatilities, it carries no scale of its own -- a reading of
    +0.3 means the same thing for a utility and for a semiconductor stock,
    which is what a single pooled model across 231 tickers requires.

    Expressed as ``log(short) - log(long)`` rather than ``short / long``. The
    raw ratio is bounded below at 0 and unbounded above, so "twice as
    turbulent" (2.0) and "half as turbulent" (0.5) sit at very different
    distances from the neutral value of 1. In logs they are symmetric, +0.69
    and -0.69, which is the shape a linear model can actually use.

    Both legs go through `safe_log`, so a window with zero volatility yields
    NaN rather than an infinity that would silently reach the design matrix.
    """
    return safe_log(short_vol) - safe_log(long_vol)
