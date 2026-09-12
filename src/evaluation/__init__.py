"""Evaluation protocol: metrics, baselines, splitters and the runner.

Deliberately written *before* any model. Fixing the measuring stick first
removes the temptation to tune the experiment until the number looks good.

Modules
-------
metrics
    Regression and classification metrics implemented from scratch.
splitters
    Expanding-window walk-forward splits with an embargo gap.
baselines
    The naive competitors every model must beat to be worth anything.
runner
    Orchestrates scale -> fit -> predict -> score across all folds.
"""
