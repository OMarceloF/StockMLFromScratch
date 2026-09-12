"""From-scratch estimators, built with NumPy only.

Every estimator here follows the scikit-learn convention (``fit`` / ``predict``)
so that it can be swapped one-for-one with its reference counterpart in the
comparison experiments.

Modules
-------
base
    Shared estimator interface and input validation.
scaler
    Standardization fitted on training data only.
linear_regression
    Ordinary least squares via the normal equation and via gradient descent,
    plus L2 (ridge) regularization.
logistic_regression
    Binary logistic regression trained by gradient descent on the cross-entropy
    loss, with optional L2 regularization.
"""
