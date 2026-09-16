"""Tests for `src.evaluation.baselines`.

Two of these are worth singling out.

`test_mean_baseline_scores_exactly_zero` closes a loop: the out-of-sample R^2 is
*defined* as improvement over the training mean, so the baseline that predicts
that mean must score exactly 0.0. If it ever does not, the metric and the
baseline have drifted apart and every comparison in the project is off.

`test_fitted_persistence_beats_naive_persistence` pins down why both variants
exist. Volatility mean-reverts, so the optimal slope on past volatility is well
below 1 and the naive rule overshoots. A model that only beats the naive version
has not shown much.
"""

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

from src.evaluation.baselines import (
    FittedPersistenceBaseline,
    MajorityClassBaseline,
    MeanBaseline,
    PersistenceBaseline,
)
from src.evaluation.classification import majority_class_accuracy, roc_auc
from src.evaluation.classification import accuracy as accuracy_score
from src.evaluation.metrics import r_squared


@pytest.fixture
def mean_reverting():
    """A target that reverts: y = 0.6 * x + noise, so the best slope is 0.6."""
    rng = np.random.default_rng(42)
    x_train = rng.normal(-1.3, 0.5, 8_000)
    y_train = 0.6 * x_train - 0.5 + rng.normal(0, 0.3, 8_000)
    x_test = rng.normal(-1.3, 0.5, 2_000)
    y_test = 0.6 * x_test - 0.5 + rng.normal(0, 0.3, 2_000)
    return x_train, y_train, x_test, y_test


class TestMeanBaseline:
    def test_predicts_the_training_mean_everywhere(self):
        baseline = MeanBaseline().fit([1.0, 2.0, 3.0, 4.0])
        assert baseline.predict(5) == pytest.approx([2.5] * 5)

    def test_scores_exactly_zero_out_of_sample(self, mean_reverting):
        """The loop closes: R^2(oos) is defined against this prediction."""
        x_train, y_train, _, y_test = mean_reverting
        baseline = MeanBaseline().fit(y_train)
        prediction = baseline.predict(len(y_test))
        assert r_squared(y_test, prediction, reference=baseline.mean_) == pytest.approx(0.0)

    def test_is_not_zero_against_the_self_reference(self, mean_reverting):
        """Against the test period's own mean it scores negative whenever the
        two periods differ -- which is the honest reading of 'the historical
        average was wrong this year'."""
        _, y_train, _, y_test = mean_reverting
        baseline = MeanBaseline().fit(y_train)
        assert r_squared(y_test, baseline.predict(len(y_test)), reference="self") <= 0

    def test_refuses_to_predict_before_fitting(self):
        with pytest.raises(RuntimeError, match="fit"):
            MeanBaseline().predict(3)


class TestPersistenceBaseline:
    def test_returns_its_input_unchanged(self):
        x = [0.3, -1.2, 4.5]
        assert PersistenceBaseline().predict(x) == pytest.approx(x)

    def test_needs_no_fitting(self):
        """Slope pinned at 1, intercept at 0 -- there is nothing to learn."""
        assert PersistenceBaseline().predict([1.0, 2.0]) == pytest.approx([1.0, 2.0])

    def test_fit_is_accepted_for_interface_compatibility(self):
        baseline = PersistenceBaseline().fit([1.0], [2.0])
        assert baseline.predict([7.0]) == pytest.approx([7.0])

    def test_scores_near_one_on_a_nearly_static_series(self):
        """Experiment 0 in miniature: when today already answers the question,
        the naive rule is almost perfect and a model cannot add anything."""
        rng = np.random.default_rng(0)
        price = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, 5_000)))
        today, tomorrow = price[:-1], price[1:]
        r2 = r_squared(tomorrow, PersistenceBaseline().predict(today), reference="self")
        assert r2 > 0.99

    def test_rejects_non_finite_input(self):
        with pytest.raises(ValueError, match="NaN"):
            PersistenceBaseline().predict([1.0, np.nan])


class TestFittedPersistenceBaseline:
    def test_matches_sklearn(self, mean_reverting):
        x_train, y_train, x_test, _ = mean_reverting
        ours = FittedPersistenceBaseline().fit(x_train, y_train)
        theirs = LinearRegression().fit(x_train.reshape(-1, 1), y_train)
        assert ours.slope_ == pytest.approx(float(theirs.coef_[0]), rel=1e-10)
        assert ours.intercept_ == pytest.approx(float(theirs.intercept_), rel=1e-10)
        assert ours.predict(x_test) == pytest.approx(
            theirs.predict(x_test.reshape(-1, 1)), rel=1e-10
        )

    def test_recovers_the_generating_slope(self, mean_reverting):
        x_train, y_train, _, _ = mean_reverting
        baseline = FittedPersistenceBaseline().fit(x_train, y_train)
        assert baseline.slope_ == pytest.approx(0.6, abs=0.03)

    def test_in_sample_r2_equals_squared_correlation(self, mean_reverting):
        """True for any single-predictor least squares fit. This is where the
        0.4785 quoted for past-volatility comes from."""
        x_train, y_train, _, _ = mean_reverting
        baseline = FittedPersistenceBaseline().fit(x_train, y_train)
        r2 = r_squared(y_train, baseline.predict(x_train), reference="self")
        assert r2 == pytest.approx(np.corrcoef(x_train, y_train)[0, 1] ** 2, rel=1e-10)

    def test_beats_naive_persistence_when_the_target_reverts(self, mean_reverting):
        """Why both variants are reported.

        The naive rule assumes slope 1; the truth here is 0.6, so it overshoots
        every prediction. Quoting only the naive baseline would flatter the model.
        """
        x_train, y_train, x_test, y_test = mean_reverting
        fitted = FittedPersistenceBaseline().fit(x_train, y_train)
        reference = float(np.mean(y_train))

        r2_fitted = r_squared(y_test, fitted.predict(x_test), reference=reference)
        r2_naive = r_squared(
            y_test, PersistenceBaseline().predict(x_test), reference=reference
        )
        assert r2_fitted > r2_naive

    def test_reduces_to_persistence_when_the_slope_really_is_one(self):
        rng = np.random.default_rng(1)
        x = rng.normal(0, 1, 5_000)
        y = x + rng.normal(0, 0.05, 5_000)
        fitted = FittedPersistenceBaseline().fit(x, y)
        assert fitted.slope_ == pytest.approx(1.0, abs=0.01)
        assert fitted.intercept_ == pytest.approx(0.0, abs=0.01)

    def test_rejects_mismatched_shapes(self):
        with pytest.raises(ValueError, match="shape mismatch"):
            FittedPersistenceBaseline().fit([1.0, 2.0, 3.0], [1.0, 2.0])

    def test_refuses_to_predict_before_fitting(self):
        with pytest.raises(RuntimeError, match="fit"):
            FittedPersistenceBaseline().predict([1.0])


class TestMajorityClassBaseline:
    @pytest.fixture
    def labels(self):
        rng = np.random.default_rng(42)
        return (rng.random(20_000) < 0.58).astype(float)

    def test_learns_the_training_base_rate(self, labels):
        baseline = MajorityClassBaseline().fit(labels)
        assert baseline.base_rate_ == pytest.approx(0.58, abs=0.02)
        assert baseline.majority_class_ == 1.0

    def test_accuracy_equals_the_majority_class_rate(self, labels):
        """The number Experiment 3 has to beat, reproduced from the other side."""
        baseline = MajorityClassBaseline().fit(labels)
        prediction = baseline.predict(len(labels))
        assert accuracy_score(labels, prediction) == pytest.approx(
            majority_class_accuracy(labels)
        )

    def test_picks_the_down_class_when_that_is_the_majority(self):
        baseline = MajorityClassBaseline().fit([0.0] * 70 + [1.0] * 30)
        assert baseline.majority_class_ == 0.0
        assert baseline.predict(4) == pytest.approx([0.0] * 4)

    def test_probabilities_are_the_base_rate_not_a_hard_label(self, labels):
        baseline = MajorityClassBaseline().fit(labels)
        probabilities = baseline.predict_proba(10)
        assert probabilities == pytest.approx([baseline.base_rate_] * 10)

    def test_constant_probabilities_give_an_auc_of_exactly_half(self, labels):
        """Every row lands in one tie, so there is no ordering at all.

        This is why AUC is the fair headline for Experiment 3: the trivial rule
        cannot score above chance on it, however lopsided the classes are.
        """
        baseline = MajorityClassBaseline().fit(labels)
        assert roc_auc(labels, baseline.predict_proba(len(labels))) == pytest.approx(0.5)

    def test_rejects_non_binary_labels(self):
        with pytest.raises(ValueError, match="only 0 and 1"):
            MajorityClassBaseline().fit([0.0, 1.0, 2.0])

    def test_refuses_to_predict_before_fitting(self):
        with pytest.raises(RuntimeError, match="fit"):
            MajorityClassBaseline().predict(3)
