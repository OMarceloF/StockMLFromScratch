"""Single source of truth for paths and constants.

Every magic number in this project lives here, with the reasoning that produced
it. The alternative -- a `21` typed into a feature module and a different `21`
typed into a notebook -- is how a pipeline silently starts training on one
horizon and evaluating on another.

This module is deliberately **declarative**: constants and paths only, no logic
and no I/O. Anything that reads a file belongs in `src.data.loader`.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# Derived from this file's own location rather than hardcoded, so the project
# works unchanged on any machine and from any working directory -- including a
# notebook running out of notebooks/ and pytest running out of the repo root.

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SAMPLE_DIR = DATA_DIR / "sample"

RAW_CSV = RAW_DIR / "master_stock_data.csv"
SAMPLE_CSV = SAMPLE_DIR / "sample_stock_data.csv"
PROCESSED_PARQUET = PROCESSED_DIR / "prices.parquet"
FEATURES_PARQUET = PROCESSED_DIR / "features.parquet"

REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# ---------------------------------------------------------------------------
# Universe selection
# ---------------------------------------------------------------------------

#: Observations before this date are discarded. The dataset reaches back to
#: 1962, but market microstructure of that era (wide spreads, fractional
#: pricing until 2001, far lower liquidity) is different enough that pooling it
#: with modern data adds noise rather than signal.
START_DATE = "1990-01-01"

#: Minimum number of rows a ticker must have to enter the study, in trading
#: days -- roughly three years. A ticker needs WARMUP rows just to produce its
#: first usable feature vector; below this threshold what remains is too short
#: to appear in more than one walk-forward fold.
MIN_HISTORY_DAYS = 756

#: Conventional number of US trading days in a year. Used to annualise
#: volatility (daily sigma * sqrt(252)).
TRADING_DAYS_PER_YEAR = 252

# ---------------------------------------------------------------------------
# Prediction horizons
# ---------------------------------------------------------------------------

#: Main forecast horizon in trading days (~1 calendar month). Targets look at
#: t+1 .. t+HORIZON; features look at t and earlier. Never the other way round.
HORIZON = 21

#: Secondary horizon, used only to check that conclusions are not an artefact
#: of the 21-day choice.
HORIZON_SENSITIVITY = 5

# ---------------------------------------------------------------------------
# Feature construction
# ---------------------------------------------------------------------------

#: Trailing windows, in trading days, over which cumulative return is measured:
#: one day, one week, one month, one quarter, half a year. Roughly logarithmic
#: spacing -- each window is ~3x the previous -- so five features span two
#: orders of magnitude of horizon without any two describing the same thing.
MOMENTUM_WINDOWS = (1, 5, 21, 63, 126)

#: Longest lookback window used by any feature (the 126-day momentum and the
#: rolling statistics built on it). `test_config.py` asserts this stays >= every
#: window actually in use.
MAX_FEATURE_WINDOW = 126

#: Rows to drop at the start of each ticker, where rolling windows are not yet
#: full and would otherwise emit values computed from partial history.
WARMUP = MAX_FEATURE_WINDOW

# ---------------------------------------------------------------------------
# Walk-forward validation
# ---------------------------------------------------------------------------

#: Trading days discarded between the end of a training fold and the start of
#: the test fold.
#:
#: This is the subtlest guard in the project. A target dated `t` is built from
#: returns over t+1..t+HORIZON, so the last HORIZON rows of any training set
#: already contain information from the test period. Without this gap the model
#: is scored partly on data it was trained on -- leakage that inflates every
#: metric. The gap must therefore be at least HORIZON; `test_config.py` asserts it.
EMBARGO = HORIZON

#: First year used as an out-of-sample test fold. Leaves 1990-2004 (15 years)
#: as the initial training window.
FIRST_TEST_YEAR = 2005

#: Last year with data (the dataset ends 2026-08-31, so this fold is partial).
LAST_TEST_YEAR = 2026

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

#: Seed for every stochastic step (shuffling in the leakage audit, mini-batch
#: ordering). The models themselves are deterministic; this exists so that the
#: few random pieces produce identical output across runs.
RANDOM_SEED = 42
