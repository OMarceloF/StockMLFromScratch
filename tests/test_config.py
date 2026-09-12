"""Guard the invariants that hold between configuration constants.

These tests deliberately assert *relationships*, never values. `assert HORIZON == 21`
would only restate the constant and would fail on every intentional change --
it tests nothing. What is worth protecting is the set of conditions that must
survive whenever someone tunes one of these numbers, because breaking them
produces results that look fine and are wrong.
"""

from src import config


class TestPaths:
    def test_root_is_the_repository_root(self):
        """ROOT is derived from __file__; if the file moves, this catches it."""
        assert (config.ROOT / "pyproject.toml").is_file()
        assert (config.ROOT / "src" / "config.py").is_file()

    def test_data_directories_exist(self):
        for path in (config.DATA_DIR, config.RAW_DIR, config.PROCESSED_DIR, config.SAMPLE_DIR):
            assert path.is_dir(), f"missing directory: {path}"

    def test_sample_dataset_is_present(self):
        """The sample is versioned, so a fresh clone must always have it."""
        assert config.SAMPLE_CSV.is_file()

    def test_paths_stay_inside_the_repository(self):
        """A stray '..' in a path constant would write outside the project."""
        for path in (config.RAW_CSV, config.SAMPLE_CSV, config.PROCESSED_PARQUET,
                     config.FEATURES_PARQUET, config.FIGURES_DIR):
            assert config.ROOT in path.parents


class TestLeakageInvariants:
    def test_embargo_covers_the_forecast_horizon(self):
        """The core anti-leakage guarantee of the whole project.

        A target dated `t` is computed from t+1..t+HORIZON. The final HORIZON
        rows of a training fold therefore already encode the beginning of the
        test fold. If the embargo were ever set below HORIZON, every metric in
        the project would be optimistically biased -- and nothing would crash
        to tell us.
        """
        assert config.EMBARGO >= config.HORIZON

    def test_embargo_covers_the_sensitivity_horizon_too(self):
        assert config.EMBARGO >= config.HORIZON_SENSITIVITY


class TestFeasibility:
    def test_surviving_tickers_can_produce_usable_rows(self):
        """A ticker that passes MIN_HISTORY_DAYS must yield at least one row.

        Each ticker loses WARMUP rows at the start (rolling windows not yet
        full) and HORIZON rows at the end (incomplete target). If the minimum
        history did not exceed their sum, the cleaning filter would keep
        tickers that contribute nothing but overhead.
        """
        assert config.MIN_HISTORY_DAYS > config.WARMUP + config.HORIZON

    def test_warmup_covers_the_longest_feature_window(self):
        assert config.WARMUP >= config.MAX_FEATURE_WINDOW


class TestWalkForward:
    def test_training_data_exists_before_the_first_test_fold(self):
        start_year = int(config.START_DATE[:4])
        assert config.FIRST_TEST_YEAR > start_year

    def test_fold_range_is_non_empty(self):
        assert config.FIRST_TEST_YEAR <= config.LAST_TEST_YEAR

    def test_initial_training_window_is_substantial(self):
        """Guards against someone shrinking the burn-in to a year or two."""
        start_year = int(config.START_DATE[:4])
        assert config.FIRST_TEST_YEAR - start_year >= 10
