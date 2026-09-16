"""Regression metrics, implemented from first principles.

Two reasons these are written out rather than imported.

The first is the project's thesis: a repository that claims to build models from
scratch and then scores them with someone else's `r2_score` has outsourced the
half that decides what the answer means.

The second is more practical, and only becomes visible once you write the
formula down. R^2 is ``1 - SS_res / SS_tot``, and `SS_tot` is the error of a
reference prediction -- but *which* reference? Calling `sklearn.metrics.r2_score`
silently answers "the mean of the values being scored". On this dataset, under
an otherwise identical walk-forward protocol, that convention reports 0.4204
where the alternative reports 0.5511. Thirteen points of difference, decided by
a default nobody chose.

So `r_squared` here has no default. The caller states the reference, and the two
legitimate answers are named:

    reference="self"        the mean of `y_true`. The textbook definition:
                            "fraction of this sample's variance explained".
    reference=<float>       a value fixed in advance -- in walk-forward, the
                            training mean. This is the out-of-sample R^2 used in
                            the forecasting literature, and it asks the question
                            a practitioner actually faces: did the model beat
                            the historical average I could have predicted?

Both are defensible; picking silently is not.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

ArrayLike = np.ndarray | list[float]


def _validate(y_true: ArrayLike, y_pred: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """Coerce to float arrays and reject anything that would score silently wrong.

    NaN is refused rather than dropped. A metric that quietly skips missing
    predictions reports a number computed on a subset the caller did not choose,
    and the row count never appears in the output to give it away.
    """
    true = np.asarray(y_true, dtype=float).ravel()
    pred = np.asarray(y_pred, dtype=float).ravel()

    if true.shape != pred.shape:
        raise ValueError(f"shape mismatch: y_true {true.shape} vs y_pred {pred.shape}")
    if true.size == 0:
        raise ValueError("cannot score an empty array")
    if not np.isfinite(true).all():
        raise ValueError("y_true contains NaN or infinity")
    if not np.isfinite(pred).all():
        raise ValueError("y_pred contains NaN or infinity")

    return true, pred


def mean_squared_error(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Average squared error -- the quantity least squares minimises."""
    true, pred = _validate(y_true, y_pred)
    return float(np.mean((true - pred) ** 2))


def root_mean_squared_error(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """RMSE, in the same units as the target."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mean_absolute_error(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Average absolute error.

    Reported beside RMSE because the two disagree in a way that is informative
    here. Squaring gives a 6-sigma day thirty-six times the weight of a
    1-sigma day, and this dataset has thousands of 6-sigma days (excess
    kurtosis 33). A large gap between RMSE and MAE means the score is being
    driven by a handful of extreme observations.
    """
    true, pred = _validate(y_true, y_pred)
    return float(np.mean(np.abs(true - pred)))


def median_absolute_error(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Median absolute error -- the typical miss, immune to the tails.

    With a 50% breakdown point this is unmoved by however many crisis months
    the fold contains. When RMSE deteriorates and this does not, the model has
    not got worse at ordinary days; the period simply had more extreme ones.
    """
    true, pred = _validate(y_true, y_pred)
    return float(np.median(np.abs(true - pred)))


def r_squared(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    reference: float | Literal["self"],
) -> float:
    """Fraction of variation explained, relative to an explicit reference.

    ``1 - SS_res / SS_tot``, where ``SS_tot`` is the squared error of always
    predicting `reference`.

    Parameters
    ----------
    reference
        ``"self"`` uses the mean of `y_true`, matching `sklearn.metrics.r2_score`.
        A float fixes the reference in advance -- pass the training mean for the
        out-of-sample R^2 that a walk-forward protocol calls for.

        There is no default on purpose. The two answers differ by 0.13 on this
        dataset, and a metric should not decide that for its caller.

    Raises
    ------
    ValueError
        If the reference explains the data perfectly (``SS_tot == 0``), which
        makes the ratio 0/0. Returning NaN here would let a degenerate fold pass
        through a mean() and quietly poison an aggregate.
    """
    true, pred = _validate(y_true, y_pred)
    baseline = float(np.mean(true)) if reference == "self" else float(reference)

    ss_res = float(np.sum((true - pred) ** 2))
    ss_tot = float(np.sum((true - baseline) ** 2))

    if ss_tot == 0.0:
        raise ValueError(
            "R^2 is undefined: every value equals the reference, so there is "
            "no variation to explain"
        )
    return 1.0 - ss_res / ss_tot


@dataclass(frozen=True)
class RegressionScores:
    """Every regression metric for one set of predictions."""

    n: int
    rmse: float
    mae: float
    median_ae: float
    r2_oos: float
    r2_self: float

    @property
    def tail_sensitivity(self) -> float:
        """RMSE divided by MAE.

        About 1.25 for normally distributed errors. Materially higher means the
        squared-error score is being carried by a few extreme misses, and the
        median is the more honest summary of a typical day.
        """
        return self.rmse / self.mae if self.mae else float("nan")

    def __str__(self) -> str:
        return (
            f"n={self.n:,}  rmse={self.rmse:.4f}  mae={self.mae:.4f}  "
            f"median_ae={self.median_ae:.4f}  R2(oos)={self.r2_oos:.4f}  "
            f"R2(self)={self.r2_self:.4f}  rmse/mae={self.tail_sensitivity:.2f}"
        )


def regression_scores(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    reference: float,
) -> RegressionScores:
    """Score one fold on every metric, both R^2 conventions included.

    Reporting the two R^2 values side by side removes the temptation to quote
    whichever is larger. `reference` is the training mean of the fold.
    """
    true, pred = _validate(y_true, y_pred)
    return RegressionScores(
        n=true.size,
        rmse=root_mean_squared_error(true, pred),
        mae=mean_absolute_error(true, pred),
        median_ae=median_absolute_error(true, pred),
        r2_oos=r_squared(true, pred, reference=reference),
        r2_self=r_squared(true, pred, reference="self"),
    )
