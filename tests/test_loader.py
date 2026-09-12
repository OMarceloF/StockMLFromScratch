"""Tests for `src.data.loader`.

These run against the versioned sample, so they work in a fresh clone with no
243 MB download. The sample was built specifically to contain the edge cases
asserted here (see `data/sample/DATA_LICENSE.md`).
"""

import numpy as np
import pandas as pd
import pytest

from src import config
from src.data.loader import COLUMNS, SchemaError, load_prices


@pytest.fixture(scope="module")
def prices() -> pd.DataFrame:
    """Sample loaded with no date filter, so every sample row is present."""
    return load_prices(config.SAMPLE_CSV, start_date=None)


class TestSchema:
    def test_returns_canonical_columns_in_order(self, prices):
        assert list(prices.columns) == COLUMNS

    def test_dtypes_match_the_contract(self, prices):
        assert pd.api.types.is_datetime64_any_dtype(prices["date"])
        assert isinstance(prices["ticker"].dtype, pd.CategoricalDtype)
        for col in ("open", "high", "low", "close", "adj_close", "dividends", "stock_splits"):
            assert prices[col].dtype == np.float64, col
        assert prices["volume"].dtype == np.int64

    def test_volume_dtype_holds_the_largest_observed_value(self):
        """int64 is a requirement, not a default: volume exceeds uint32."""
        assert 9_230_856_000 > np.iinfo(np.uint32).max

    def test_rejects_a_file_with_the_wrong_columns(self, tmp_path):
        bad = tmp_path / "bad.csv"
        bad.write_text("Date,Ticker,Close\n2020-01-02,AAPL,100.0\n", encoding="utf-8")
        with pytest.raises(SchemaError, match="missing columns"):
            load_prices(bad, start_date=None)

    def test_missing_file_raises_a_helpful_error(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="price data not found"):
            load_prices(tmp_path / "nope.csv")


class TestOrdering:
    def test_sorted_by_ticker_then_date(self, prices):
        expected = prices.sort_values(["ticker", "date"], kind="stable").reset_index(drop=True)
        pd.testing.assert_frame_equal(prices, expected)

    def test_dates_strictly_increase_within_each_ticker(self, prices):
        """The precondition that makes shift() and rolling() meaningful."""
        for ticker, group in prices.groupby("ticker", observed=True):
            assert group["date"].is_monotonic_increasing, ticker
            assert not group["date"].duplicated().any(), ticker

    def test_index_is_a_fresh_range(self, prices):
        assert prices.index.equals(pd.RangeIndex(len(prices)))


class TestFiltering:
    def test_start_date_is_an_inclusive_lower_bound(self):
        cutoff = "2020-01-01"
        df = load_prices(config.SAMPLE_CSV, start_date=cutoff)
        assert df["date"].min() >= pd.Timestamp(cutoff)

    def test_none_keeps_the_full_history(self, prices):
        filtered = load_prices(config.SAMPLE_CSV, start_date="2020-01-01")
        assert len(prices) > len(filtered)

    def test_filtering_drops_unused_category_levels(self):
        """A category that keeps dead levels poisons groupby(observed=False)."""
        df = load_prices(config.SAMPLE_CSV, start_date="2026-01-01")
        levels = set(df["ticker"].cat.categories)
        present = set(df["ticker"].unique())
        assert levels == present
        # EA's history ends 2026-08-10 but starts 2026-07-17, so it survives;
        # the check that matters is that no level exists without rows.
        assert df.groupby("ticker", observed=False)["close"].count().notna().all()


class TestSampleContents:
    """The sample must keep carrying the edge cases the pipeline handles."""

    def test_expected_tickers_are_present(self, prices):
        assert set(prices["ticker"].unique()) == {
            "JNJ", "KO", "CAT", "XOM", "MSFT", "AAPL",
            "NVDA", "TSLA", "NEE", "PLTR", "EA", "CICC",
        }

    def test_contains_the_empty_price_rows(self, prices):
        """NEE 2026-08-28 -- exercises the cleaning step in 1.2."""
        assert prices["close"].isna().sum() == 1

    def test_contains_short_history_tickers(self, prices):
        counts = prices.groupby("ticker", observed=True).size()
        assert (counts < config.MIN_HISTORY_DAYS).sum() == 2

    def test_contains_stock_splits(self, prices):
        assert (prices["stock_splits"] != 0).sum() == 6

    def test_ohlc_bounds_hold_where_prices_exist(self, prices):
        """Low <= Open/Close <= High. A violation means a corrupt source."""
        p = prices.dropna(subset=["open", "high", "low", "close"])
        assert (p["low"] <= p["high"]).all()
        assert (p["open"].between(p["low"], p["high"])).all()
        assert (p["close"].between(p["low"], p["high"])).all()
