"""Feature engineering and target construction.

The golden rule enforced throughout this subpackage: a feature dated ``t`` may
only use information available up to and including ``t``; a target dated ``t``
looks at ``t+1 .. t+HORIZON``. Any violation is leakage.

Modules
-------
technical
    Scale-free predictors (returns, volatility, volume z-scores, price ratios).
targets
    Forward-looking targets for the four experiments.
pipeline
    Assembles the final feature matrix and trims warm-up / incomplete rows.
"""
