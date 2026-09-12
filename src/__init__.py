"""Stock ML From Scratch.

Linear and logistic regression implemented from first principles with NumPy,
validated against scikit-learn, and evaluated on ~1.3M rows of daily US equity
data under a walk-forward protocol.

Subpackages
-----------
data
    Loading and cleaning the raw price history.
features
    Leakage-free technical features and forward-looking targets.
models
    From-scratch estimators (scaler, linear regression, logistic regression).
evaluation
    Metrics, baselines, walk-forward splitters and the experiment runner.
viz
    Shared plotting helpers used by the notebooks.
"""

__version__ = "0.1.0"
