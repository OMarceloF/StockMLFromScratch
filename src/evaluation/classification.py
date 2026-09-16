"""Classification metrics, implemented from first principles.

The trap this module is built around is accuracy. Experiment 3 predicts whether
the next 21 days gain, and on this dataset **57.95% of those windows are
positive**. A classifier reporting 55% accuracy looks like it has beaten a coin
flip; it has in fact lost to a one-line rule that answers "up" every time.

So `accuracy` is never returned alone here. `ClassificationScores` carries the
majority-class rate beside it and the difference between them, because that
difference is the only part that represents work done.

`balanced_accuracy` is reported for the same reason from the other direction:
it averages the two per-class rates, so the always-up rule scores exactly 0.5
no matter how lopsided the classes are.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

ArrayLike = np.ndarray | list[float]


def _validate_labels(y_true: ArrayLike) -> np.ndarray:
    labels = np.asarray(y_true, dtype=float).ravel()
    if labels.size == 0:
        raise ValueError("cannot score an empty array")
    if not np.isfinite(labels).all():
        raise ValueError("y_true contains NaN or infinity")
    if not np.isin(labels, (0.0, 1.0)).all():
        raise ValueError("y_true must contain only 0 and 1")
    return labels


def _validate_pair(y_true: ArrayLike, other: ArrayLike, name: str) -> tuple[np.ndarray, np.ndarray]:
    labels = _validate_labels(y_true)
    values = np.asarray(other, dtype=float).ravel()
    if labels.shape != values.shape:
        raise ValueError(f"shape mismatch: y_true {labels.shape} vs {name} {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} contains NaN or infinity")
    return labels, values


@dataclass(frozen=True)
class ConfusionMatrix:
    """The four counts every other classification metric is built from."""

    tn: int
    fp: int
    fn: int
    tp: int

    @property
    def n(self) -> int:
        return self.tn + self.fp + self.fn + self.tp

    def __str__(self) -> str:
        return (
            "            predicted\n"
            "             down    up\n"
            f"  actual down {self.tn:>6,}{self.fp:>6,}\n"
            f"         up   {self.fn:>6,}{self.tp:>6,}"
        )


def confusion_matrix(y_true: ArrayLike, y_pred: ArrayLike) -> ConfusionMatrix:
    """Counts of the four outcomes, for hard 0/1 predictions."""
    labels, predictions = _validate_pair(y_true, y_pred, "y_pred")
    if not np.isin(predictions, (0.0, 1.0)).all():
        raise ValueError("y_pred must contain only 0 and 1; threshold scores first")

    positive = labels == 1.0
    predicted_positive = predictions == 1.0
    return ConfusionMatrix(
        tn=int(np.sum(~positive & ~predicted_positive)),
        fp=int(np.sum(~positive & predicted_positive)),
        fn=int(np.sum(positive & ~predicted_positive)),
        tp=int(np.sum(positive & predicted_positive)),
    )


def _ratio(numerator: int, denominator: int) -> float:
    """Undefined ratios return NaN rather than a substituted zero.

    A classifier that predicts no positives has undefined precision: it made no
    positive claims, so none of them were wrong. Reporting 0.0 there says "every
    positive call missed", which is a different and false statement.
    """
    return float(numerator) / denominator if denominator else float("nan")


def accuracy(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Fraction correct. Meaningless without `majority_class_accuracy` beside it."""
    cm = confusion_matrix(y_true, y_pred)
    return _ratio(cm.tp + cm.tn, cm.n)


def precision(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Of the days called up, how many rose."""
    cm = confusion_matrix(y_true, y_pred)
    return _ratio(cm.tp, cm.tp + cm.fp)


def recall(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Of the days that rose, how many were called. Also the true positive rate."""
    cm = confusion_matrix(y_true, y_pred)
    return _ratio(cm.tp, cm.tp + cm.fn)


def specificity(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Of the days that fell, how many were called. The true negative rate."""
    cm = confusion_matrix(y_true, y_pred)
    return _ratio(cm.tn, cm.tn + cm.fp)


def f1_score(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Harmonic mean of precision and recall.

    Harmonic rather than arithmetic so that a model cannot buy a good score by
    maximising one at the other's expense: predicting "up" always gives recall
    1.0, and the harmonic mean drags the result back down to the precision.
    """
    p, r = precision(y_true, y_pred), recall(y_true, y_pred)
    if not np.isfinite(p) or not np.isfinite(r) or (p + r) == 0:
        return float("nan")
    return 2 * p * r / (p + r)


def balanced_accuracy(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Mean of recall and specificity -- accuracy with the class sizes removed.

    The always-up rule scores exactly 0.5 here regardless of the base rate,
    which is what makes this the fair headline number for Experiment 3.
    """
    r, s = recall(y_true, y_pred), specificity(y_true, y_pred)
    return float(np.mean([r, s]))


def base_rate(y_true: ArrayLike) -> float:
    """Proportion of positives. 0.5795 on this dataset -- not 0.5."""
    return float(np.mean(_validate_labels(y_true)))


def majority_class_accuracy(y_true: ArrayLike) -> float:
    """What always answering with the more common class would score.

    The opponent. Any accuracy at or below this represents no work.
    """
    rate = base_rate(y_true)
    return max(rate, 1.0 - rate)


def brier_score(y_true: ArrayLike, y_prob: ArrayLike) -> float:
    """Mean squared error of the predicted probabilities.

    Measures calibration as well as discrimination: a model that says 0.9 and
    is right 90% of the time scores better than one that says 0.6 and is right
    90% of the time, even though both rank identically and share an AUC.
    """
    labels, probabilities = _validate_pair(y_true, y_prob, "y_prob")
    return float(np.mean((probabilities - labels) ** 2))


def roc_curve(y_true: ArrayLike, y_score: ArrayLike) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """False positive rate, true positive rate and threshold at every cut point.

    Built by sorting the scores from highest to lowest and walking down the
    list. At each step one more observation is called positive, so the running
    count of true and false positives *is* the curve -- no thresholds need to be
    guessed, because the only ones that change anything are the observed scores
    themselves.

    Tied scores collapse to a single point. Splitting them would let the curve
    depend on the order the data happened to arrive in.

    Returns
    -------
    (fpr, tpr, thresholds)
        Each starting at the (0, 0) corner, where nothing is called positive.
    """
    labels, scores = _validate_pair(y_true, y_score, "y_score")

    positives = labels.sum()
    negatives = labels.size - positives
    if positives == 0 or negatives == 0:
        raise ValueError("an ROC curve needs both classes present")

    # mergesort is stable, so ties keep a deterministic order.
    order = np.argsort(-scores, kind="mergesort")
    sorted_labels = labels[order]
    sorted_scores = scores[order]

    true_positives = np.cumsum(sorted_labels)
    false_positives = np.cumsum(1.0 - sorted_labels)

    # Keep only the last index of each run of equal scores.
    last_of_run = np.r_[np.flatnonzero(np.diff(sorted_scores)), labels.size - 1]

    tpr = np.r_[0.0, true_positives[last_of_run] / positives]
    fpr = np.r_[0.0, false_positives[last_of_run] / negatives]
    thresholds = np.r_[np.inf, sorted_scores[last_of_run]]
    return fpr, tpr, thresholds


def roc_auc(y_true: ArrayLike, y_score: ArrayLike) -> float:
    """Area under the ROC curve, by trapezoidal integration.

    Threshold-free and unmoved by class imbalance, which is what makes it the
    right headline for Experiment 3: 0.5 is chance whether the base rate is 50%
    or 58%.

    It also has an interpretation worth holding on to. The area equals the
    probability that a randomly chosen rising day is scored above a randomly
    chosen falling one -- which means it can be computed a completely different
    way, from the ranks of the positives (the Mann-Whitney statistic).
    `test_classification.py` checks the two derivations against each other.
    """
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(np.trapezoid(tpr, fpr))


@dataclass(frozen=True)
class ClassificationScores:
    """Every classification metric, with the baseline it must be read against."""

    n: int
    threshold: float
    base_rate: float
    majority_accuracy: float
    accuracy: float
    balanced_accuracy: float
    precision: float
    recall: float
    specificity: float
    f1: float
    roc_auc: float
    brier: float
    confusion: ConfusionMatrix

    @property
    def lift_over_majority(self) -> float:
        """Accuracy minus what answering with the majority class would score.

        The only part of the accuracy figure that represents work. Negative
        means the model is worse than the one-line rule.
        """
        return self.accuracy - self.majority_accuracy

    def __str__(self) -> str:
        return (
            f"n={self.n:,}  base_rate={self.base_rate:.4f}\n"
            f"  accuracy={self.accuracy:.4f}  majority={self.majority_accuracy:.4f}  "
            f"lift={self.lift_over_majority:+.4f}\n"
            f"  balanced_accuracy={self.balanced_accuracy:.4f}  auc={self.roc_auc:.4f}  "
            f"brier={self.brier:.4f}\n"
            f"  precision={self.precision:.4f}  recall={self.recall:.4f}  "
            f"specificity={self.specificity:.4f}  f1={self.f1:.4f}"
        )


def classification_scores(
    y_true: ArrayLike,
    y_prob: ArrayLike,
    *,
    threshold: float = 0.5,
) -> ClassificationScores:
    """Score one fold on every classification metric.

    Parameters
    ----------
    threshold
        Probability above which a day is called up. The 0.5 default is a
        convention, not an optimum -- with a 58% base rate a model can be well
        calibrated and still put almost every prediction above 0.5. The
        threshold-free metrics (`roc_auc`) and the imbalance-free ones
        (`balanced_accuracy`) are the ones to read first.
    """
    labels, probabilities = _validate_pair(y_true, y_prob, "y_prob")
    predictions = (probabilities > threshold).astype(float)

    return ClassificationScores(
        n=labels.size,
        threshold=threshold,
        base_rate=base_rate(labels),
        majority_accuracy=majority_class_accuracy(labels),
        accuracy=accuracy(labels, predictions),
        balanced_accuracy=balanced_accuracy(labels, predictions),
        precision=precision(labels, predictions),
        recall=recall(labels, predictions),
        specificity=specificity(labels, predictions),
        f1=f1_score(labels, predictions),
        roc_auc=roc_auc(labels, probabilities),
        brier=brier_score(labels, probabilities),
        confusion=confusion_matrix(labels, predictions),
    )
