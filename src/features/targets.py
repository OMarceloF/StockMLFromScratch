"""The four prediction targets, and the only forward-looking code in the project.

Everything in `technical.py` looks strictly backwards: a feature dated `t` uses
data up to and including `t`. This module is the mirror image -- every quantity
here is built from `t+1 .. t+horizon` and is, by construction, unknowable at `t`.

Keeping the two directions in separate modules is deliberate. Leakage happens
when a forward-looking value sneaks into the feature matrix, and the easiest way
to let that happen is to compute both kinds of quantity side by side in the same
function. Here the split is structural: if a column came from this module it is
a target, and it never belongs in `X`.

The four targets correspond to the project's four experiments:

    next_close            Experiment 0 -- the negative control. Predicting the
                          next day's price gives an R^2 near 1.0 that means
                          nothing, because the previous price is already almost
                          the whole answer.
    forward_volatility    Experiment 1 -- the main case. Volatility clusters, so
                          this one has genuine signal.
    forward_return        Experiment 2 -- the second negative control. Returns
                          have no usable autocorrelation, so R^2 lands near zero
                          and that is the honest result.
    forward_direction     Experiment 3 -- the classification task, and the
                          target for logistic regression.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import config
from src.features.technical import (
    log_returns,
    momentum,
    realized_volatility,
    safe_log,
)


def shift_backward(series: pd.Series, groups: pd.Series, horizon: int) -> pd.Series:
    """Bring a value from `horizon` rows ahead onto the current row.

    The single operation every target in this module is built from, isolated so
    it can be tested once rather than re-derived four times.

    The grouping is the whole point. A bare ``shift(-21)`` on the stacked frame
    would, at the end of each ticker's history, reach across the boundary and
    hand one company's future to another -- 231 tickers means 231 such splices,
    each one silently wrong and none of them raising anything. Grouped, those
    rows correctly become NaN.

    Note the sign: a *negative* shift pulls future rows backwards onto the
    present one. That is the correct direction for a target and the wrong
    direction for a feature, which is why nothing in `technical.py` uses it.
    """
    return series.groupby(groups, observed=True).shift(-horizon)


def forward_volatility(
    returns: pd.Series,
    groups: pd.Series,
    horizon: int = config.HORIZON,
    *,
    log: bool = True,
    annualize: bool = True,
) -> pd.Series:
    """Realised volatility over the next `horizon` trading days. **Experiment 1.**

    Built by computing trailing volatility and then shifting it backwards, so
    the value landing on row `t` is the volatility of returns `t+1 .. t+horizon`
    -- no overlap with `t` itself.

    Returned in logs by default. Volatility is strictly positive and strongly
    right-skewed (skewness 3.4 on this dataset at a 21-day window); in logs it
    is 0.44. Least squares fits a straight line and weights errors by their
    square, so a right-skewed target lets a handful of crisis months dominate
    the fit. Logs also make the target's scale symmetric: predicting 40% when
    the truth is 20% is penalised as heavily as predicting 10%, which matches
    how a volatility error actually matters.

    Measured on the full dataset, log volatility is also the better target in
    plain predictive terms: a single trailing-volatility predictor reaches
    R^2 0.4785 in logs against 0.4108 raw.
    """
    trailing = realized_volatility(returns, groups, horizon, annualize=annualize)
    forward = shift_backward(trailing, groups, horizon)
    return safe_log(forward) if log else forward


def forward_return(
    returns: pd.Series,
    groups: pd.Series,
    horizon: int = config.HORIZON,
) -> pd.Series:
    """Cumulative log return over the next `horizon` trading days. **Experiment 2.**

    The sum of returns `t+1 .. t+horizon`, which is exactly `log(P_{t+horizon} /
    P_t)` -- the additivity property that made log returns the right choice in
    the first place.

    Worth being precise about what that does and does not touch. The target
    reads the **price** at `t`, because that is the price the position would be
    opened at and there is nothing unknowable about it. It reads no **return**
    at or before `t`, and those are what the features are made of. Perturbing
    any price strictly before `t` leaves this value unchanged; `test_targets.py`
    pins that down.

    This target is expected to be close to unpredictable. The EDA measured a
    correlation of -0.016 between today's return and tomorrow's, an R^2 of
    0.00025. Keeping it as a first-class target rather than quietly dropping it
    is the point of Experiment 2: a reported null result is a result.
    """
    trailing = momentum(returns, groups, horizon)
    return shift_backward(trailing, groups, horizon)


def forward_direction(
    returns: pd.Series,
    groups: pd.Series,
    horizon: int = config.HORIZON,
) -> pd.Series:
    """1 if the next `horizon` days gain, 0 if they lose. **Experiment 3.**

    The sign of `forward_return`, kept as float rather than int so that rows
    with no target stay NaN. An integer column cannot hold NaN, and casting
    early would turn "unknown" into 0 -- a silent relabelling of every ticker's
    final rows as losses.

    The base rate is **not** 50%. Equities drift upward, so a majority of
    21-day windows are positive, and the classifier's real opponent is a rule
    that always predicts "up". Phase 3.4 builds that baseline explicitly,
    because a model scoring 55% against a 56% base rate is worse than useless
    while looking like a coin-flip beater.
    """
    forward = forward_return(returns, groups, horizon)
    return (forward > 0).astype(float).where(forward.notna())


def next_close(close: pd.Series, groups: pd.Series) -> pd.Series:
    """Tomorrow's closing price. **Experiment 0 -- the negative control.**

    Deliberately a raw price, not a return: reproducing the mistake this project
    exists to dismantle requires making it faithfully. A model fitted on lagged
    prices will predict this with R^2 near 1.0, and that number will be
    worthless, because `close` at `t` is already almost the entire answer.

    Experiment 0's job is to show that the naive "tomorrow equals today"
    baseline matches or beats the fitted model, and that the fitted
    coefficient on the previous price lands near 1.0 while everything else
    lands near 0 -- the model announcing, if anyone reads it, that it learned
    nothing.
    """
    return shift_backward(close, groups, 1)


#: Column names produced by `build_targets`, in the order the experiments run.
TARGET_COLUMNS = [
    "y_next_close",
    "y_log_vol_fwd",
    "y_ret_fwd",
    "y_dir_fwd",
]


def build_targets(
    df: pd.DataFrame,
    *,
    horizon: int = config.HORIZON,
    returns: pd.Series | None = None,
) -> pd.DataFrame:
    """All four targets for a canonical price frame.

    Parameters
    ----------
    df
        Canonical frame from `src.data.loader.load_prices`.
    horizon
        Forecast horizon in trading days.
    returns
        Pre-computed log returns, to avoid recomputing them when the caller
        already has them. Computed from `df` when omitted.

    Returns
    -------
    DataFrame
        Columns `TARGET_COLUMNS`, aligned to `df.index`. The last `horizon`
        rows of every ticker are NaN -- their future has not happened yet.
    """
    groups = df["ticker"]
    if returns is None:
        returns = log_returns(df)

    return pd.DataFrame(
        {
            "y_next_close": next_close(df["close"], groups),
            "y_log_vol_fwd": forward_volatility(returns, groups, horizon),
            "y_ret_fwd": forward_return(returns, groups, horizon),
            "y_dir_fwd": forward_direction(returns, groups, horizon),
        },
        index=df.index,
    )
