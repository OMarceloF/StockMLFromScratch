"""Cross-cutting checks on every metric at once.

`test_metrics.py` and `test_classification.py` check each function against one
fixture. That proves the formula is right; it does not prove it stays right at
other sample sizes, other class balances, other magnitudes, or near the awkward
boundaries. This file covers the gap three ways:

**Randomised parity.** The same sklearn comparison, swept across many
configurations rather than one -- small samples, extreme imbalance, near-perfect
and near-useless predictions.

**Mathematical properties.** Statements that must hold for *any* input, so they
need no reference implementation to compare against: MAE never exceeds RMSE, an
ROC area and its mirror sum to one, the confusion matrix always accounts for
every row. These catch a class of bug that agreeing with sklearn would not --
namely, both being wrong the same way.

**Numerical stability.** The metrics run on log-volatility around -1.3 and on
raw prices up to 4,354. Scale invariance is a property they need to actually
have, not one to assume.
"""

import numpy as np
import pytest
from sklearn import metrics as sk

from src.evaluation import classification as cls
from src.evaluation import metrics as reg

# (n, noise, signal) -- from an almost perfect fit to an almost useless one.
REGRESSION_CASES = [
    (50, 0.1, 1.0),
    (500, 1.0, 1.0),
    (5_000, 3.0, 0.2),
    (20_000, 0.01, 1.0),
    (200, 5.0, 0.0),
]

# (n, base_rate, separation) -- balanced through to heavily skewed.
CLASSIFICATION_CASES = [
    (200, 0.50, 0.40),
    (2_000, 0.58, 0.18),
    (5_000, 0.90, 0.30),
    (5_000, 0.10, 0.05),
    (1_000, 0.58, 0.00),
]


def regression_sample(seed: int, n: int, noise: float, signal: float):
    rng = np.random.default_rng(seed)
    y = rng.normal(0, 2, n)
    return y, signal * y + rng.normal(0, noise, n)


def classification_sample(seed: int, n: int, rate: float, separation: float):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < rate).astype(float)
    scores = np.clip(0.5 + separation * (y - 0.5) + rng.normal(0, 0.15, n), 1e-4, 1 - 1e-4)
    return y, scores


@pytest.mark.parametrize("case", REGRESSION_CASES, ids=lambda c: f"n{c[0]}_noise{c[1]}")
@pytest.mark.parametrize("seed", [0, 1, 2])
class TestRegressionParity:
    def test_every_regression_metric_matches_sklearn(self, seed, case):
        y, p = regression_sample(seed, *case)
        assert reg.mean_squared_error(y, p) == pytest.approx(
            sk.mean_squared_error(y, p), rel=1e-12
        )
        assert reg.root_mean_squared_error(y, p) == pytest.approx(
            sk.root_mean_squared_error(y, p), rel=1e-12
        )
        assert reg.mean_absolute_error(y, p) == pytest.approx(
            sk.mean_absolute_error(y, p), rel=1e-12
        )
        assert reg.median_absolute_error(y, p) == pytest.approx(
            sk.median_absolute_error(y, p), rel=1e-12
        )
        assert reg.r_squared(y, p, reference="self") == pytest.approx(
            sk.r2_score(y, p), rel=1e-12
        )


@pytest.mark.parametrize("case", CLASSIFICATION_CASES, ids=lambda c: f"n{c[0]}_rate{c[1]}")
@pytest.mark.parametrize("seed", [0, 1, 2])
class TestClassificationParity:
    def test_every_classification_metric_matches_sklearn(self, seed, case):
        y, s = classification_sample(seed, *case)
        if len(np.unique(y)) < 2:
            pytest.skip("degenerate draw: only one class present")
        p = (s > 0.5).astype(float)

        assert cls.accuracy(y, p) == pytest.approx(sk.accuracy_score(y, p), rel=1e-12)
        assert cls.brier_score(y, s) == pytest.approx(
            sk.brier_score_loss(y, s), rel=1e-12
        )
        assert cls.roc_auc(y, s) == pytest.approx(sk.roc_auc_score(y, s), rel=1e-12)
        assert cls.balanced_accuracy(y, p) == pytest.approx(
            sk.balanced_accuracy_score(y, p), rel=1e-12
        )

        if p.sum() > 0:
            assert cls.precision(y, p) == pytest.approx(
                sk.precision_score(y, p), rel=1e-12
            )
            assert cls.f1_score(y, p) == pytest.approx(sk.f1_score(y, p), rel=1e-12)
        if y.sum() > 0:
            assert cls.recall(y, p) == pytest.approx(sk.recall_score(y, p), rel=1e-12)

    def test_specificity_matches_recall_of_the_negative_class(self, seed, case):
        """The one metric with no direct sklearn function.

        Specificity is recall with the roles of the classes swapped, which
        sklearn expresses as `pos_label=0`.
        """
        y, s = classification_sample(seed, *case)
        if len(np.unique(y)) < 2:
            pytest.skip("degenerate draw: only one class present")
        p = (s > 0.5).astype(float)
        if (1 - y).sum() == 0:
            pytest.skip("no negatives to score")
        assert cls.specificity(y, p) == pytest.approx(
            sk.recall_score(y, p, pos_label=0), rel=1e-12
        )


@pytest.mark.parametrize("seed", range(8))
class TestMathematicalProperties:
    """Statements true for any input -- no reference implementation involved.

    These catch the failure mode parity tests cannot: our metric and sklearn's
    agreeing because both are wrong in the same way.
    """

    def test_mae_never_exceeds_rmse(self, seed):
        """Jensen's inequality. Equality only when every error has one magnitude."""
        y, p = regression_sample(seed, 2_000, 1.0, 0.8)
        assert reg.mean_absolute_error(y, p) <= reg.root_mean_squared_error(y, p) + 1e-12

    def test_r2_never_exceeds_one(self, seed):
        y, p = regression_sample(seed, 2_000, 1.0, 0.8)
        assert reg.r_squared(y, p, reference="self") <= 1.0

    def test_self_reference_gives_the_smallest_r2(self, seed):
        """The sample mean minimises SS_tot, so it maximises SS_res/SS_tot."""
        y, p = regression_sample(seed, 2_000, 1.0, 0.8)
        strictest = reg.r_squared(y, p, reference="self")
        for elsewhere in (-5.0, 0.0, 5.0):
            assert reg.r_squared(y, p, reference=elsewhere) >= strictest - 1e-12

    def test_mse_is_zero_only_for_exact_predictions(self, seed):
        y, p = regression_sample(seed, 500, 1.0, 0.8)
        assert reg.mean_squared_error(y, y) == 0.0
        assert reg.mean_squared_error(y, p) > 0.0

    def test_confusion_matrix_accounts_for_every_row(self, seed):
        y, s = classification_sample(seed, 2_000, 0.58, 0.2)
        cm = cls.confusion_matrix(y, (s > 0.5).astype(float))
        assert cm.n == len(y)
        assert cm.tp + cm.fn == int(y.sum())
        assert cm.tn + cm.fp == int((1 - y).sum())

    def test_rates_stay_within_zero_and_one(self, seed):
        y, s = classification_sample(seed, 2_000, 0.58, 0.2)
        p = (s > 0.5).astype(float)
        for value in (
            cls.accuracy(y, p), cls.precision(y, p), cls.recall(y, p),
            cls.specificity(y, p), cls.f1_score(y, p),
            cls.balanced_accuracy(y, p), cls.roc_auc(y, s), cls.brier_score(y, s),
        ):
            assert 0.0 <= value <= 1.0

    def test_reversing_the_scores_mirrors_the_auc(self, seed):
        """AUC(s) + AUC(-s) == 1: reversing the ranking reflects the curve."""
        y, s = classification_sample(seed, 2_000, 0.58, 0.2)
        assert cls.roc_auc(y, s) + cls.roc_auc(y, -s) == pytest.approx(1.0)

    def test_auc_reads_ranks_not_values(self, seed):
        """Any strictly increasing transform of the scores leaves AUC alone."""
        y, s = classification_sample(seed, 2_000, 0.58, 0.2)
        for transform in (lambda v: v * 1000, lambda v: np.exp(v), lambda v: v**3):
            assert cls.roc_auc(y, transform(s)) == pytest.approx(cls.roc_auc(y, s))

    def test_f1_lies_between_precision_and_recall(self, seed):
        y, s = classification_sample(seed, 2_000, 0.58, 0.2)
        p = (s > 0.5).astype(float)
        precision, recall = cls.precision(y, p), cls.recall(y, p)
        assert min(precision, recall) <= cls.f1_score(y, p) <= max(precision, recall)


class TestNumericalStability:
    """The metrics run on log volatility near -1.3 and on prices up to 4,354."""

    @pytest.mark.parametrize("scale", [1e-6, 1.0, 1e3, 1e6])
    def test_r2_is_invariant_to_rescaling(self, scale):
        y, p = regression_sample(0, 5_000, 1.0, 0.8)
        assert reg.r_squared(y * scale, p * scale, reference="self") == pytest.approx(
            reg.r_squared(y, p, reference="self"), rel=1e-9
        )

    @pytest.mark.parametrize("shift", [-1e6, -1e3, 0.0, 1e6])
    def test_r2_is_invariant_to_shifting(self, shift):
        y, p = regression_sample(0, 5_000, 1.0, 0.8)
        assert reg.r_squared(y + shift, p + shift, reference="self") == pytest.approx(
            reg.r_squared(y, p, reference="self"), rel=1e-7
        )

    @pytest.mark.parametrize("scale", [1e-6, 1e3, 1e6])
    def test_rmse_scales_linearly(self, scale):
        y, p = regression_sample(0, 5_000, 1.0, 0.8)
        assert reg.root_mean_squared_error(y * scale, p * scale) == pytest.approx(
            reg.root_mean_squared_error(y, p) * scale, rel=1e-9
        )

    def test_survives_a_target_with_almost_no_variance(self):
        rng = np.random.default_rng(0)
        y = 1.0 + rng.normal(0, 1e-9, 1_000)
        p = y + rng.normal(0, 1e-10, 1_000)
        assert reg.r_squared(y, p, reference="self") > 0.9

    def test_auc_survives_scores_crushed_against_the_boundaries(self):
        """A confident classifier pushes probabilities to 0 and 1; the ranking
        must survive the loss of resolution."""
        rng = np.random.default_rng(0)
        y = (rng.random(5_000) < 0.58).astype(float)
        s = np.clip(0.5 + 40 * (y - 0.5) + rng.normal(0, 1, 5_000), 1e-12, 1 - 1e-12)
        assert cls.roc_auc(y, s) == pytest.approx(sk.roc_auc_score(y, s), rel=1e-12)

    def test_small_samples_still_score(self):
        assert reg.mean_squared_error([1.0], [2.0]) == pytest.approx(1.0)
        assert cls.roc_auc([0.0, 1.0], [0.2, 0.8]) == pytest.approx(1.0)


class TestCoverageGuard:
    """Fails when a metric is added without being wired into the parity sweep.

    Not ceremony: the sweep is the only place that exercises every metric at
    once, so a function missing from it is a function whose behaviour outside
    its single fixture is unknown.
    """

    REGRESSION_METRICS = {
        "mean_squared_error", "root_mean_squared_error", "mean_absolute_error",
        "median_absolute_error", "r_squared", "regression_scores",
    }
    CLASSIFICATION_METRICS = {
        "confusion_matrix", "accuracy", "precision", "recall", "specificity",
        "f1_score", "balanced_accuracy", "base_rate", "majority_class_accuracy",
        "brier_score", "roc_curve", "roc_auc", "classification_scores",
    }

    @staticmethod
    def public_functions(module) -> set[str]:
        import inspect

        return {
            name
            for name, obj in vars(module).items()
            if not name.startswith("_")
            and inspect.isfunction(obj)
            and obj.__module__ == module.__name__
        }

    def test_no_regression_metric_is_unlisted(self):
        assert self.public_functions(reg) == self.REGRESSION_METRICS

    def test_no_classification_metric_is_unlisted(self):
        assert self.public_functions(cls) == self.CLASSIFICATION_METRICS
