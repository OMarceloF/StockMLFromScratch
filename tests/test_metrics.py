"""Tests for `src.evaluation.metrics`.

Every metric is checked twice: against a hand-computable case, and against
scikit-learn on random data. The first proves the formula is the one intended;
the second proves it agrees with the reference implementation the rest of the
world uses. A metric that only matched itself would be a private convention
dressed up as a standard.

This is also the first appearance of the pattern Phase 4 depends on -- our
implementation versus sklearn's, `np.allclose` deciding.
"""

import numpy as np
import pytest
from sklearn import metrics as sk

from src.evaluation.metrics import (
    RegressionScores,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r_squared,
    regression_scores,
    root_mean_squared_error,
)


@pytest.fixture
def random_pair():
    rng = np.random.default_rng(42)
    y_true = rng.normal(0, 2, 5_000)
    y_pred = y_true * 0.8 + rng.normal(0, 1, 5_000)
    return y_true, y_pred


class TestAgainstSklearn:
    def test_mse_matches(self, random_pair):
        y, p = random_pair
        assert mean_squared_error(y, p) == pytest.approx(
            sk.mean_squared_error(y, p), rel=1e-12
        )

    def test_rmse_matches(self, random_pair):
        y, p = random_pair
        assert root_mean_squared_error(y, p) == pytest.approx(
            sk.root_mean_squared_error(y, p), rel=1e-12
        )

    def test_mae_matches(self, random_pair):
        y, p = random_pair
        assert mean_absolute_error(y, p) == pytest.approx(
            sk.mean_absolute_error(y, p), rel=1e-12
        )

    def test_median_ae_matches(self, random_pair):
        y, p = random_pair
        assert median_absolute_error(y, p) == pytest.approx(
            sk.median_absolute_error(y, p), rel=1e-12
        )

    def test_r2_with_self_reference_matches(self, random_pair):
        """sklearn's r2_score is exactly our reference="self" case."""
        y, p = random_pair
        assert r_squared(y, p, reference="self") == pytest.approx(
            sk.r2_score(y, p), rel=1e-12
        )


class TestHandComputable:
    def test_mse_on_a_worked_example(self):
        # errors: 1, -1, 2  ->  squares 1, 1, 4  ->  mean 2
        assert mean_squared_error([1, 2, 3], [0, 3, 1]) == pytest.approx(2.0)

    def test_mae_on_a_worked_example(self):
        assert mean_absolute_error([1, 2, 3], [0, 3, 1]) == pytest.approx(4 / 3)

    def test_median_ae_on_a_worked_example(self):
        # absolute errors 1, 1, 2 -> median 1
        assert median_absolute_error([1, 2, 3], [0, 3, 1]) == pytest.approx(1.0)

    def test_perfect_predictions_score_one(self):
        y = [1.0, 5.0, 9.0]
        assert r_squared(y, y, reference="self") == pytest.approx(1.0)
        assert mean_squared_error(y, y) == 0.0

    def test_predicting_the_mean_scores_zero(self):
        y = [1.0, 5.0, 9.0]
        assert r_squared(y, [5.0] * 3, reference="self") == pytest.approx(0.0)

    def test_r2_can_go_negative(self):
        """A model worse than the reference gets a negative score -- and must,
        because a floor at zero would hide a model that hurts."""
        assert r_squared([1.0, 2.0, 3.0], [10.0, 10.0, 10.0], reference="self") < 0


class TestTheReferenceChoice:
    """The decision this module exists to make explicit."""

    def test_reference_is_required(self):
        with pytest.raises(TypeError):
            r_squared([1.0, 2.0], [1.0, 2.0])  # type: ignore[call-arg]

    def test_the_two_conventions_disagree(self):
        """Not an edge case -- a 0.13 gap on the real dataset."""
        y = np.array([2.0, 4.0, 6.0, 8.0])
        p = np.array([3.0, 4.0, 5.0, 9.0])
        assert r_squared(y, p, reference="self") != pytest.approx(
            r_squared(y, p, reference=0.0)
        )

    def test_self_reference_is_the_strictest_possible(self):
        """`reference="self"` always gives the *smallest* R^2 of any reference.

        The sample mean is the value that minimises SS_tot, and R^2 is
        `1 - SS_res/SS_tot`, so a smaller denominator means a smaller score.
        Any reference fixed elsewhere inflates SS_tot and therefore R^2.

        That is precisely why the out-of-sample convention reports the higher
        number here (0.5512 against 0.4204), and why picking whichever looks
        better after the fact is not a neutral act.
        """
        y = np.array([2.0, 4.0, 6.0, 8.0])
        p = np.array([3.0, 4.0, 5.0, 9.0])
        assert r_squared(y, p, reference=0.0) > r_squared(y, p, reference="self")

    def test_a_shifted_reference_changes_only_the_denominator(self):
        y = np.array([1.0, 2.0, 3.0, 4.0])
        p = np.array([1.5, 2.5, 2.5, 3.5])
        ss_res = float(np.sum((y - p) ** 2))
        for ref in (0.0, 2.5, 10.0):
            ss_tot = float(np.sum((y - ref) ** 2))
            assert r_squared(y, p, reference=ref) == pytest.approx(1 - ss_res / ss_tot)

    def test_undefined_r2_raises_instead_of_returning_nan(self):
        """A degenerate fold must stop the run, not slip through a mean()."""
        with pytest.raises(ValueError, match="undefined"):
            r_squared([3.0, 3.0, 3.0], [3.0, 3.0, 3.0], reference="self")


class TestValidation:
    def test_rejects_mismatched_shapes(self):
        with pytest.raises(ValueError, match="shape mismatch"):
            mean_squared_error([1, 2, 3], [1, 2])

    def test_rejects_empty_input(self):
        with pytest.raises(ValueError, match="empty"):
            mean_squared_error([], [])

    def test_rejects_nan_in_truth(self):
        """Silently dropping NaN would score a subset the caller never chose."""
        with pytest.raises(ValueError, match="y_true contains NaN"):
            mean_squared_error([1.0, np.nan], [1.0, 1.0])

    def test_rejects_nan_in_predictions(self):
        with pytest.raises(ValueError, match="y_pred contains NaN"):
            mean_squared_error([1.0, 1.0], [1.0, np.nan])

    def test_rejects_infinity(self):
        with pytest.raises(ValueError, match="infinity"):
            mean_squared_error([1.0, 1.0], [1.0, np.inf])

    def test_accepts_lists_and_arrays_alike(self):
        assert mean_squared_error([1, 2], [1, 2]) == mean_squared_error(
            np.array([1.0, 2.0]), np.array([1.0, 2.0])
        )

    def test_flattens_column_vectors(self):
        column = np.array([[1.0], [2.0], [3.0]])
        assert mean_squared_error(column, [1.0, 2.0, 3.0]) == 0.0


class TestRegressionScores:
    def test_carries_every_metric(self, random_pair):
        y, p = random_pair
        scores = regression_scores(y, p, reference=float(np.mean(y)) + 0.3)
        assert isinstance(scores, RegressionScores)
        assert scores.n == len(y)
        assert scores.rmse == pytest.approx(root_mean_squared_error(y, p))
        assert scores.r2_self == pytest.approx(r_squared(y, p, reference="self"))

    def test_reports_both_r2_conventions(self, random_pair):
        """Side by side, so neither can be quoted in isolation."""
        y, p = random_pair
        scores = regression_scores(y, p, reference=1.5)
        assert scores.r2_oos != pytest.approx(scores.r2_self)

    def test_tail_sensitivity_is_near_one_point_two_five_for_normal_errors(self):
        rng = np.random.default_rng(0)
        y = rng.normal(0, 1, 200_000)
        p = y + rng.normal(0, 0.5, 200_000)
        scores = regression_scores(y, p, reference=0.0)
        assert scores.tail_sensitivity == pytest.approx(1.2533, abs=0.02)

    def test_tail_sensitivity_rises_with_fat_tails(self):
        """The diagnostic working: heavy-tailed errors push RMSE away from MAE.

        Same target both times, so only the error distribution differs.
        """
        rng = np.random.default_rng(0)
        n = 200_000
        y = rng.normal(0, 3, n)
        light = regression_scores(y, y + rng.normal(0, 1, n), reference=0.0)
        heavy = regression_scores(y, y + rng.standard_t(2.5, n), reference=0.0)
        assert heavy.tail_sensitivity > light.tail_sensitivity * 1.2

    def test_renders(self, random_pair):
        y, p = random_pair
        text = str(regression_scores(y, p, reference=0.0))
        assert "rmse=" in text and "R2(oos)=" in text and "R2(self)=" in text
