"""Tests for `src.evaluation.classification`.

Same discipline as the regression metrics: every function checked against a
hand-computable case and against scikit-learn.

The AUC gets a third check that the others do not. It is computed here by
integrating the ROC curve, and there is a completely independent way to arrive
at the same number -- the Mann-Whitney statistic, built from the ranks of the
positive scores with no curve involved at all. Two derivations that share no
code agreeing to twelve decimals is much stronger evidence than either one
matching a reference implementation.
"""

import numpy as np
import pytest
from sklearn import metrics as sk

from src.evaluation.classification import (
    accuracy,
    balanced_accuracy,
    base_rate,
    brier_score,
    classification_scores,
    confusion_matrix,
    f1_score,
    majority_class_accuracy,
    precision,
    recall,
    roc_auc,
    roc_curve,
    specificity,
)


@pytest.fixture
def imbalanced():
    """Labels with a 58% base rate, mirroring the real target."""
    rng = np.random.default_rng(42)
    n = 20_000
    labels = (rng.random(n) < 0.58).astype(float)
    scores = np.clip(0.5 + 0.18 * (labels - 0.5) + rng.normal(0, 0.15, n), 0.001, 0.999)
    return labels, scores


class TestConfusionMatrix:
    def test_counts_a_worked_example(self):
        y = [0, 0, 1, 1, 1]
        p = [0, 1, 0, 1, 1]
        cm = confusion_matrix(y, p)
        assert (cm.tn, cm.fp, cm.fn, cm.tp) == (1, 1, 1, 2)
        assert cm.n == 5

    def test_matches_sklearn(self, imbalanced):
        y, s = imbalanced
        p = (s > 0.5).astype(float)
        cm = confusion_matrix(y, p)
        assert np.array_equal(
            np.array([[cm.tn, cm.fp], [cm.fn, cm.tp]]), sk.confusion_matrix(y, p)
        )

    def test_rejects_probabilities_passed_as_predictions(self):
        with pytest.raises(ValueError, match="threshold scores first"):
            confusion_matrix([0, 1], [0.3, 0.7])

    def test_renders(self, imbalanced):
        y, s = imbalanced
        text = str(confusion_matrix(y, (s > 0.5).astype(float)))
        assert "predicted" in text and "actual" in text


class TestPointMetrics:
    def test_worked_example(self):
        y = [0, 0, 1, 1, 1]
        p = [0, 1, 0, 1, 1]
        assert accuracy(y, p) == pytest.approx(3 / 5)
        assert precision(y, p) == pytest.approx(2 / 3)
        assert recall(y, p) == pytest.approx(2 / 3)
        assert specificity(y, p) == pytest.approx(1 / 2)
        assert f1_score(y, p) == pytest.approx(2 / 3)
        assert balanced_accuracy(y, p) == pytest.approx((2 / 3 + 1 / 2) / 2)

    def test_match_sklearn(self, imbalanced):
        y, s = imbalanced
        p = (s > 0.5).astype(float)
        assert accuracy(y, p) == pytest.approx(sk.accuracy_score(y, p), rel=1e-12)
        assert precision(y, p) == pytest.approx(sk.precision_score(y, p), rel=1e-12)
        assert recall(y, p) == pytest.approx(sk.recall_score(y, p), rel=1e-12)
        assert f1_score(y, p) == pytest.approx(sk.f1_score(y, p), rel=1e-12)
        assert balanced_accuracy(y, p) == pytest.approx(
            sk.balanced_accuracy_score(y, p), rel=1e-12
        )

    def test_brier_matches_sklearn(self, imbalanced):
        y, s = imbalanced
        assert brier_score(y, s) == pytest.approx(sk.brier_score_loss(y, s), rel=1e-12)

    def test_undefined_precision_is_nan_not_zero(self):
        """No positive calls made means none of them were wrong.

        sklearn substitutes 0.0, which reads as "every positive call missed" --
        a different and false claim.
        """
        assert np.isnan(precision([0, 1, 1], [0, 0, 0]))
        assert np.isnan(f1_score([0, 1, 1], [0, 0, 0]))


class TestTheBaseRateTrap:
    """The reason this module exists."""

    def test_base_rate_is_not_half(self, imbalanced):
        y, _ = imbalanced
        assert base_rate(y) == pytest.approx(0.58, abs=0.02)

    def test_always_up_scores_the_base_rate_on_accuracy(self, imbalanced):
        y, _ = imbalanced
        always_up = np.ones_like(y)
        assert accuracy(y, always_up) == pytest.approx(base_rate(y))
        assert majority_class_accuracy(y) == pytest.approx(base_rate(y))

    def test_always_up_scores_exactly_half_on_balanced_accuracy(self, imbalanced):
        """Why balanced accuracy is the fair headline: the trivial rule cannot
        hide behind the class sizes."""
        y, _ = imbalanced
        assert balanced_accuracy(y, np.ones_like(y)) == pytest.approx(0.5)

    def test_a_55_percent_model_has_negative_lift(self, imbalanced):
        """The exact failure this project is guarding against."""
        y, _ = imbalanced
        rng = np.random.default_rng(7)
        weak = np.where(rng.random(len(y)) < 0.55, y, 1 - y)  # ~55% correct
        scores = classification_scores(y, weak.astype(float))
        assert 0.5 < scores.accuracy < scores.majority_accuracy
        assert scores.lift_over_majority < 0

    def test_majority_accuracy_handles_either_dominant_class(self):
        assert majority_class_accuracy([1, 1, 1, 0]) == pytest.approx(0.75)
        assert majority_class_accuracy([0, 0, 0, 1]) == pytest.approx(0.75)


class TestRocCurve:
    def test_matches_sklearn(self, imbalanced):
        """Compared against `drop_intermediate=False`, which is the full curve.

        sklearn drops points lying on a straight segment by default -- 7,081
        points instead of 19,938 on this fixture. That is a plotting
        optimisation: the area is identical either way. Our version keeps every
        cut point, which is the faithful curve and the one that makes the
        threshold column meaningful.
        """
        y, s = imbalanced
        fpr, tpr, _ = roc_curve(y, s)
        sk_fpr, sk_tpr, _ = sk.roc_curve(y, s, drop_intermediate=False)
        assert fpr == pytest.approx(sk_fpr)
        assert tpr == pytest.approx(sk_tpr)

    def test_dropping_intermediate_points_would_not_change_the_area(self, imbalanced):
        """Why the difference above is cosmetic."""
        y, s = imbalanced
        sk_fpr, sk_tpr, _ = sk.roc_curve(y, s)  # default: points dropped
        assert np.trapezoid(sk_tpr, sk_fpr) == pytest.approx(roc_auc(y, s), rel=1e-12)

    def test_starts_at_the_origin_and_ends_at_one_one(self, imbalanced):
        y, s = imbalanced
        fpr, tpr, _ = roc_curve(y, s)
        assert (fpr[0], tpr[0]) == (0.0, 0.0)
        assert (fpr[-1], tpr[-1]) == pytest.approx((1.0, 1.0))

    def test_both_rates_increase_monotonically(self, imbalanced):
        y, s = imbalanced
        fpr, tpr, _ = roc_curve(y, s)
        assert np.all(np.diff(fpr) >= 0) and np.all(np.diff(tpr) >= 0)

    def test_tied_scores_collapse_to_one_point(self):
        """Otherwise the curve would depend on the order rows arrived in."""
        y = [0, 1, 0, 1]
        s = [0.5, 0.5, 0.5, 0.5]
        fpr, _, _ = roc_curve(y, s)
        assert len(fpr) == 2  # the origin plus a single cut

    def test_a_perfect_ranking_reaches_the_corner(self):
        y = [0, 0, 1, 1]
        s = [0.1, 0.2, 0.8, 0.9]
        fpr, tpr, _ = roc_curve(y, s)
        assert 1.0 in tpr[fpr == 0.0]

    def test_requires_both_classes(self):
        with pytest.raises(ValueError, match="both classes"):
            roc_curve([1, 1, 1], [0.2, 0.5, 0.9])


class TestRocAuc:
    def test_matches_sklearn(self, imbalanced):
        y, s = imbalanced
        assert roc_auc(y, s) == pytest.approx(sk.roc_auc_score(y, s), rel=1e-12)

    def test_matches_the_mann_whitney_statistic(self, imbalanced):
        """The independent derivation.

        AUC is the probability a random positive outranks a random negative, so
        it can be read straight off the ranks with no curve, no thresholds and
        no integration. Agreement between the two routes is the strongest check
        available for this function.
        """
        y, s = imbalanced
        ranks = np.argsort(np.argsort(s, kind="mergesort"), kind="mergesort") + 1.0
        # average ranks within ties, as the statistic requires
        order = np.argsort(s, kind="mergesort")
        sorted_scores = s[order]
        start = 0
        for end in np.r_[np.flatnonzero(np.diff(sorted_scores)) + 1, len(s)]:
            ranks[order[start:end]] = np.mean(ranks[order[start:end]])
            start = end

        n_pos = int(y.sum())
        n_neg = len(y) - n_pos
        u = ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2
        assert roc_auc(y, s) == pytest.approx(u / (n_pos * n_neg), rel=1e-12)

    def test_random_scores_sit_near_one_half(self):
        rng = np.random.default_rng(3)
        y = (rng.random(50_000) < 0.58).astype(float)
        assert roc_auc(y, rng.random(50_000)) == pytest.approx(0.5, abs=0.01)

    def test_perfect_ranking_scores_one(self):
        y = np.r_[np.zeros(100), np.ones(100)]
        assert roc_auc(y, np.arange(200.0)) == pytest.approx(1.0)

    def test_inverted_ranking_scores_zero(self):
        y = np.r_[np.zeros(100), np.ones(100)]
        assert roc_auc(y, np.arange(200.0)[::-1]) == pytest.approx(0.0)

    def test_is_unmoved_by_class_imbalance(self):
        """Why AUC is the headline for Experiment 3 and accuracy is not.

        The same ranking quality, at a 50% and at a 90% base rate, scores the
        same. Accuracy would not.
        """
        rng = np.random.default_rng(11)

        def auc_at(rate):
            n = 40_000
            y = (rng.random(n) < rate).astype(float)
            s = 0.5 + 0.2 * (y - 0.5) + rng.normal(0, 0.15, n)
            return roc_auc(y, s)

        assert auc_at(0.5) == pytest.approx(auc_at(0.9), abs=0.02)

    def test_is_invariant_to_monotone_rescaling(self):
        """AUC reads ranks, not values -- so calibration cannot change it."""
        y, s = (np.r_[np.zeros(500), np.ones(500)], None)
        rng = np.random.default_rng(5)
        s = 0.5 + 0.2 * (y - 0.5) + rng.normal(0, 0.2, 1000)
        assert roc_auc(y, s) == pytest.approx(roc_auc(y, 1 / (1 + np.exp(-3 * s))))


class TestClassificationScores:
    def test_carries_the_baseline_beside_the_accuracy(self, imbalanced):
        y, s = imbalanced
        scores = classification_scores(y, s)
        assert scores.majority_accuracy == pytest.approx(majority_class_accuracy(y))
        assert scores.lift_over_majority == pytest.approx(
            scores.accuracy - scores.majority_accuracy
        )

    def test_threshold_is_configurable(self, imbalanced):
        y, s = imbalanced
        low = classification_scores(y, s, threshold=0.4)
        high = classification_scores(y, s, threshold=0.6)
        assert low.recall > high.recall
        assert low.specificity < high.specificity

    def test_auc_does_not_depend_on_the_threshold(self, imbalanced):
        y, s = imbalanced
        a = classification_scores(y, s, threshold=0.3)
        b = classification_scores(y, s, threshold=0.7)
        assert a.roc_auc == pytest.approx(b.roc_auc)

    def test_renders_with_the_baseline_visible(self, imbalanced):
        y, s = imbalanced
        text = str(classification_scores(y, s))
        assert "majority=" in text and "lift=" in text and "auc=" in text


class TestValidation:
    def test_rejects_non_binary_labels(self):
        with pytest.raises(ValueError, match="only 0 and 1"):
            accuracy([0, 1, 2], [0, 1, 1])

    def test_rejects_nan_labels(self):
        with pytest.raises(ValueError, match="NaN"):
            base_rate([0.0, np.nan])

    def test_rejects_shape_mismatch(self):
        with pytest.raises(ValueError, match="shape mismatch"):
            accuracy([0, 1, 1], [0, 1])

    def test_rejects_empty_input(self):
        with pytest.raises(ValueError, match="empty"):
            base_rate([])
