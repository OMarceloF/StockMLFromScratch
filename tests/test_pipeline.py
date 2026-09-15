"""Tests for `src.features.pipeline`, including the leakage audit.

The audit is the most important test in this repository. Leakage does not raise,
does not crash, and does not look wrong -- it just inflates every metric the
project reports. So it is checked structurally rather than by inspection:

    Perturb every price after a cutoff date. Recompute.
    - Features at or before the cutoff must NOT move.
    - Targets near the cutoff MUST move.

The second half matters as much as the first. A "target" that failed to look
forward would sail through the first assertion while quietly making the whole
exercise meaningless.
"""

import numpy as np
import pandas as pd
import pytest

from src import config
from src.features.pipeline import (
    FEATURE_COLUMNS,
    NAIVE_PRICE_COLUMNS,
    MOMENTUM_FEATURES,
    PRICE_FEATURES,
    VOLATILITY_FEATURES,
    VOLUME_FEATURES,
    build_dataset,
    build_features,
    build_naive_price_features,
)
from src.features.targets import TARGET_COLUMNS


def synthetic_frame(n_days: int = 500, tickers: tuple[str, ...] = ("AAA", "BBB")) -> pd.DataFrame:
    """A canonical-shaped frame with plausible OHLCV, built from a fixed seed."""
    rng = np.random.default_rng(config.RANDOM_SEED)
    rows = []
    for i, ticker in enumerate(tickers):
        dates = pd.bdate_range("2015-01-01", periods=n_days)
        close = (10 + 90 * i) * np.exp(np.cumsum(rng.normal(0.0003, 0.02, n_days)))
        spread = np.abs(rng.normal(0, 0.01, n_days))
        rows += [
            {
                "date": d, "ticker": ticker,
                "open": c * (1 + rng.normal(0, 0.003)),
                "high": c * (1 + s), "low": c * (1 - s),
                "close": c, "adj_close": c,
                "volume": int(rng.lognormal(15, 0.4)),
                "dividends": 0.0, "stock_splits": 0.0,
            }
            for d, c, s in zip(dates, close, spread)
        ]
    df = pd.DataFrame(rows)
    df["ticker"] = df["ticker"].astype("category")
    return df.sort_values(["ticker", "date"]).reset_index(drop=True)


@pytest.fixture(scope="module")
def prices() -> pd.DataFrame:
    return synthetic_frame()


@pytest.fixture(scope="module")
def dataset(prices):
    return build_dataset(prices)


class TestLeakageAudit:
    """Perturb the future; the past must not notice."""

    @staticmethod
    def perturb_after(df: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
        """Scramble every price and volume strictly after `cutoff`.

        The level and the intraday spread are shocked **independently**, and
        that detail is load-bearing. Scaling high, low and close by one shared
        factor leaves `log(H/L)` untouched, so `log_park_21` -- correctly, since
        it is scale-free -- would not move at all, and the audit would pass
        vacuously for that column. `test_the_perturbation_is_strong_enough...`
        exists to catch exactly that, and did.
        """
        rng = np.random.default_rng(1234)
        out = df.copy()
        future = out["date"] > cutoff
        n = int(future.sum())

        level = rng.uniform(0.5, 2.0, n)
        spread = rng.uniform(0.3, 3.0, n)

        close = out.loc[future, "close"].to_numpy() * level
        half_range = (
            (out.loc[future, "high"].to_numpy() - out.loc[future, "low"].to_numpy())
            / 2 * level * spread
        )
        out.loc[future, "close"] = close
        out.loc[future, "adj_close"] = out.loc[future, "adj_close"].to_numpy() * level
        out.loc[future, "open"] = out.loc[future, "open"].to_numpy() * level
        out.loc[future, "high"] = close + half_range
        out.loc[future, "low"] = np.maximum(close - half_range, 0.01)

        out.loc[future, "volume"] = (
            out.loc[future, "volume"].to_numpy() * rng.uniform(0.2, 5.0, n)
        ).astype("int64")
        return out

    def test_no_feature_changes_when_the_future_is_scrambled(self, prices):
        """The audit. Any forward-looking feature fails here and nowhere else."""
        cutoff = pd.Timestamp("2016-06-30")
        original = build_features(prices)
        perturbed = build_features(self.perturb_after(prices, cutoff))

        past = prices["date"] <= cutoff
        for column in FEATURE_COLUMNS:
            pd.testing.assert_series_equal(
                original.loc[past, column],
                perturbed.loc[past, column],
                check_names=False,
                obj=f"feature {column} moved when only the future changed",
            )

    def test_the_perturbation_is_strong_enough_to_be_detectable(self, prices):
        """Guards the audit itself.

        If the perturbation were too weak, or applied to no rows, the test above
        would pass vacuously. This confirms features *after* the cutoff do move.
        """
        cutoff = pd.Timestamp("2016-06-30")
        original = build_features(prices)
        perturbed = build_features(self.perturb_after(prices, cutoff))

        future = prices["date"] > cutoff
        moved = [
            column for column in FEATURE_COLUMNS
            if not np.allclose(
                original.loc[future, column].dropna(),
                perturbed.loc[future, column].dropna(),
            )
        ]
        assert set(moved) == set(FEATURE_COLUMNS)

    def test_targets_do_move_when_the_future_is_scrambled(self, prices):
        """The mirror image. Targets that ignored the future would pass the
        first test too -- and be useless."""
        from src.features.targets import build_targets

        cutoff = pd.Timestamp("2016-06-30")
        original = build_targets(prices)
        perturbed = build_targets(self.perturb_after(prices, cutoff))

        # Rows just before the cutoff look forward across it.
        window = (prices["date"] <= cutoff) & (
            prices["date"] > cutoff - pd.Timedelta(days=20)
        )
        for column in TARGET_COLUMNS:
            a = original.loc[window, column].dropna()
            b = perturbed.loc[window, column].dropna()
            assert not np.allclose(a, b), f"target {column} ignored the future"


class TestDesignMatrix:
    def test_has_full_column_rank(self, dataset):
        """The check that would have caught `vol_ratio` automatically.

        That feature was exactly `log_vol_5 - log_vol_21`, which makes X rank
        deficient. The normal equation does not raise on it -- floating point
        leaves a non-zero determinant -- it silently returns coefficients that
        are not unique.
        """
        df, _ = dataset
        X = np.column_stack([np.ones(len(df)), df[FEATURE_COLUMNS].to_numpy()])
        assert np.linalg.matrix_rank(X) == X.shape[1]

    def test_is_well_conditioned(self, dataset):
        """float64 carries ~16 digits; a condition number above 1e8 spends half
        of them. Anything near 1e15 means the coefficients are noise."""
        df, _ = dataset
        X = np.column_stack([np.ones(len(df)), df[FEATURE_COLUMNS].to_numpy()])
        assert np.linalg.cond(X) < 1e6

    def test_contains_no_missing_or_infinite_values(self, dataset):
        df, _ = dataset
        block = df[FEATURE_COLUMNS].to_numpy()
        assert np.isfinite(block).all()

    def test_no_feature_is_constant(self, dataset):
        """A constant column is collinear with the intercept."""
        df, _ = dataset
        assert (df[FEATURE_COLUMNS].std() > 0).all()

    def test_no_feature_pair_is_near_perfectly_collinear(self, dataset):
        df, _ = dataset
        # pandas 3.0 hands back read-only arrays (copy-on-write), so ask for a
        # writable copy before zeroing the diagonal.
        corr = df[FEATURE_COLUMNS].corr().abs().to_numpy(copy=True)
        np.fill_diagonal(corr, 0.0)
        assert corr.max() < 0.99


class TestFeatureSet:
    def test_the_selected_features_are_the_documented_blocks(self):
        assert FEATURE_COLUMNS == (
            VOLATILITY_FEATURES + MOMENTUM_FEATURES + VOLUME_FEATURES + PRICE_FEATURES
        )

    def test_rejected_candidates_are_absent(self):
        """Measurement rejected these; the matrix must not quietly regain them."""
        for rejected in ("vol_ratio", "rsi_14", "pos_in_range_63", "log_dollar_vol_21"):
            assert rejected not in FEATURE_COLUMNS

    def test_no_target_column_leaks_into_the_feature_list(self):
        assert not set(FEATURE_COLUMNS) & set(TARGET_COLUMNS)
        assert not any(c.startswith("y_") for c in FEATURE_COLUMNS)

    def test_features_are_aligned_to_the_input_index(self, prices):
        assert build_features(prices).index.equals(prices.index)


class TestTrimming:
    def test_each_ticker_loses_its_warmup_and_its_tail(self, prices):
        df, _ = build_dataset(prices)
        for ticker, group in prices.groupby("ticker", observed=True):
            kept = df[df["ticker"] == ticker]
            assert kept["date"].min() >= group["date"].iloc[config.WARMUP - 1]
            assert kept["date"].max() <= group["date"].iloc[-config.HORIZON - 1]

    def test_the_report_accounts_for_what_was_dropped(self, prices):
        df, report = build_dataset(prices)
        assert report.rows_out == len(df)
        assert report.dropped_warmup == config.WARMUP * prices["ticker"].nunique()
        assert report.dropped_tail == config.HORIZON * prices["ticker"].nunique()

    def test_report_renders(self, dataset):
        text = str(dataset[1])
        assert "Dataset report" in text and "warm-up" in text

    def test_dropna_false_keeps_every_row(self, prices):
        df, _ = build_dataset(prices, dropna=False)
        assert len(df) == len(prices)
        assert df[FEATURE_COLUMNS].isna().any(axis=1).sum() > 0


class TestDatasetShape:
    def test_carries_identifiers_features_and_targets(self, dataset):
        df, _ = dataset
        assert list(df.columns) == ["date", "ticker", *FEATURE_COLUMNS, *TARGET_COLUMNS]

    def test_index_is_a_fresh_range(self, dataset):
        df, _ = dataset
        assert df.index.equals(pd.RangeIndex(len(df)))

    def test_rows_stay_sorted_by_ticker_then_date(self, dataset):
        df, _ = dataset
        expected = df.sort_values(["ticker", "date"], kind="stable").reset_index(drop=True)
        pd.testing.assert_frame_equal(df, expected)

    def test_unused_category_levels_are_dropped(self, dataset):
        df, _ = dataset
        assert set(df["ticker"].cat.categories) == set(df["ticker"].unique())


class TestNaivePriceFeatures:
    def test_produces_the_documented_lags(self, prices):
        got = build_naive_price_features(prices)
        assert list(got.columns) == NAIVE_PRICE_COLUMNS

    def test_lag_one_is_yesterdays_close(self, prices):
        got = build_naive_price_features(prices)
        aapl = prices["ticker"] == "AAA"
        close = prices.loc[aapl, "close"].to_numpy()
        assert got.loc[aapl, "close_lag_1"].to_numpy()[1:] == pytest.approx(close[:-1])

    def test_does_not_cross_the_ticker_boundary(self, prices):
        got = build_naive_price_features(prices)
        first_rows = prices.groupby("ticker", observed=True).head(1).index
        assert got.loc[first_rows, "close_lag_1"].isna().all()

    def test_is_deliberately_not_scale_free(self, prices):
        """Documents the contrast: these carry price level, unlike every other
        feature in the project. That is what makes Experiment 0 work."""
        got = build_naive_price_features(prices)
        by_ticker = got.groupby(prices["ticker"], observed=True)["close_lag_1"].median()
        assert by_ticker.max() / by_ticker.min() > 5
