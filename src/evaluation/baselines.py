"""The opponents every model has to beat.

A metric on its own says nothing. R^2 0.55 is excellent for forecasting equity
volatility and dismal for fitting a calibration curve; 58% accuracy is strong on
a balanced problem and a failure when 58% of the labels are already positive.
The number only becomes a claim once it sits beside what a trivial rule scores
on the same data, under the same protocol.

Four baselines, one per experiment:

    MeanBaseline              predicts the training mean. The floor: its
                              out-of-sample R^2 is exactly 0 by construction,
                              which is what makes that metric's zero meaningful.
    PersistenceBaseline       predicts that the next window looks like the last
                              one. No fitting at all. The honest naive rule --
                              and for Experiment 0, the rule that beats the
                              model outright.
    FittedPersistenceBaseline the same idea, but with the slope and intercept
                              fitted. The *hard* opponent, and the one a model
                              with thirteen features actually has to beat.
    MajorityClassBaseline     always answers with the more common class. The
                              opponent for Experiment 3, where it scores 58%.

Reporting only the easy baseline is a way of losing while appearing to win, so
the volatility experiments report both persistence variants.

Every baseline follows `fit` then `predict`, the same shape as the estimators in
Phase 4, so the runner can score them through identical code -- including
fitting them inside each fold, which they need as much as any model does.
"""

from __future__ import annotations

from typing import Self

import numpy as np

ArrayLike = np.ndarray | list[float]


def _as_1d(values: ArrayLike, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float).ravel()
    if array.size == 0:
        raise ValueError(f"{name} is empty")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or infinity")
    return array


class MeanBaseline:
    """Predict the training mean, whatever the features say.

    This is the reference the out-of-sample R^2 is defined against, so scoring
    it is a tautology: it returns exactly 0.0. That is worth doing anyway --
    it is a live check that the metric and the baseline agree on what "no skill"
    means, and it fails loudly if either drifts.
    """

    name = "mean"

    def __init__(self) -> None:
        self.mean_: float | None = None

    def fit(self, y_train: ArrayLike) -> Self:
        self.mean_ = float(np.mean(_as_1d(y_train, "y_train")))
        return self

    def predict(self, n_rows: int) -> np.ndarray:
        if self.mean_ is None:
            raise RuntimeError("call fit() before predict()")
        return np.full(int(n_rows), self.mean_)


class PersistenceBaseline:
    """Predict that what just happened happens again.

    For volatility: next month's realised volatility equals last month's.
    For price (Experiment 0): tomorrow's close equals today's.

    There is nothing to fit -- the slope is pinned at 1 and the intercept at 0.
    That is the point. It is the rule someone would use without a model at all,
    and on this dataset it reaches walk-forward R^2 0.31 on log volatility and
    roughly 0.9996 on raw price.

    `fit` exists only so the runner can treat it like every other estimator.
    """

    name = "persistence"

    def fit(self, x_train: ArrayLike | None = None, y_train: ArrayLike | None = None) -> Self:
        return self

    def predict(self, x: ArrayLike) -> np.ndarray:
        return _as_1d(x, "x")


class FittedPersistenceBaseline:
    """Least squares on the single most informative feature.

    The same information as `PersistenceBaseline`, but allowed to choose its own
    slope and intercept. That freedom matters more than it sounds: volatility
    mean-reverts, so the optimal slope on past volatility is well below 1, and
    the naive rule is systematically too extreme.

    This is the opponent that makes the headline honest. A thirteen-feature
    model beating "next month equals last month" has shown very little; beating
    the best possible line through that one feature is the real claim.

    For a single predictor, in-sample R^2 equals the squared correlation between
    feature and target -- 0.4785 on this dataset -- which is where the 0.48
    figure quoted throughout the notebooks comes from.
    """

    name = "fitted persistence"

    def __init__(self) -> None:
        self.intercept_: float | None = None
        self.slope_: float | None = None

    def fit(self, x_train: ArrayLike, y_train: ArrayLike) -> Self:
        x = _as_1d(x_train, "x_train")
        y = _as_1d(y_train, "y_train")
        if x.shape != y.shape:
            raise ValueError(f"shape mismatch: x {x.shape} vs y {y.shape}")

        design = np.column_stack([np.ones(x.size), x])
        coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
        self.intercept_, self.slope_ = float(coefficients[0]), float(coefficients[1])
        return self

    def predict(self, x: ArrayLike) -> np.ndarray:
        if self.slope_ is None:
            raise RuntimeError("call fit() before predict()")
        return self.intercept_ + self.slope_ * _as_1d(x, "x")


class MajorityClassBaseline:
    """Always answer with whichever class was more common in training.

    Scores 58% accuracy on this dataset's direction target, which is the number
    Experiment 3 has to beat -- not 50%.

    `predict_proba` returns the training base rate for every row rather than a
    hard 0 or 1. That makes the baseline scoreable on Brier and AUC too, and it
    exposes a useful fact: a constant probability puts every observation in one
    tie, so its AUC is exactly 0.5. A model can only earn AUC above chance by
    *ordering* days, which is a stronger claim than getting most of them right.
    """

    name = "majority class"

    def __init__(self) -> None:
        self.base_rate_: float | None = None

    def fit(self, y_train: ArrayLike) -> Self:
        y = _as_1d(y_train, "y_train")
        if not np.isin(y, (0.0, 1.0)).all():
            raise ValueError("y_train must contain only 0 and 1")
        self.base_rate_ = float(np.mean(y))
        return self

    @property
    def majority_class_(self) -> float:
        if self.base_rate_ is None:
            raise RuntimeError("call fit() before predict()")
        return 1.0 if self.base_rate_ >= 0.5 else 0.0

    def predict_proba(self, n_rows: int) -> np.ndarray:
        if self.base_rate_ is None:
            raise RuntimeError("call fit() before predict()")
        return np.full(int(n_rows), self.base_rate_)

    def predict(self, n_rows: int) -> np.ndarray:
        return np.full(int(n_rows), self.majority_class_)
