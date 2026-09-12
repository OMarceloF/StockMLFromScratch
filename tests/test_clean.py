"""Tests for `src.data.clean`.

Two kinds of test here. The first run on the real sample and assert the known
edge cases are handled. The second build tiny synthetic frames containing a
defect the sample does not have -- duplicates, negative prices, broken OHLC --
because a filter that is never exercised is a filter nobody knows works.
"""

import numpy as np
import pandas as pd
import pytest

from src import config
from src.data.clean import clean_prices
from src.data.loader import COLUMNS, load_prices


@pytest.fixture(scope="module")
def raw() -> pd.DataFrame:
    return load_prices(config.SAMPLE_CSV, start_date=None)


@pytest.fixture(scope="module")
def cleaned(raw):
    return clean_prices(raw)


def make_frame(rows: list[dict]) -> pd.DataFrame:
    """Build a minimal canonical frame, defaults filled in."""
    base = {
        "open": 10.0, "high": 11.0, "low": 9.0, "close": 10.5, "adj_close": 10.5,
        "volume": 1_000, "dividends": 0.0, "stock_splits": 0.0,
    }
    df = pd.DataFrame([{**base, **r} for r in rows])
    df["date"] = pd.to_datetime(df["date"])
    df["ticker"] = df["ticker"].astype("category")
    return df[COLUMNS]


def series(ticker: str, n: int, **overrides) -> list[dict]:
    """n consecutive business days for one ticker."""
    dates = pd.bdate_range("2000-01-03", periods=n)
    return [{"date": d, "ticker": ticker, **overrides} for d in dates]


class TestOnTheRealSample:
    def test_removes_the_empty_price_row(self, raw, cleaned):
        df, report = cleaned
        assert raw["close"].isna().sum() == 1
        assert df["close"].isna().sum() == 0
        step = next(s for s in report.steps if s.name == "missing prices")
        assert step.rows_removed == 1

    def test_drops_the_two_short_history_tickers(self, cleaned):
        df, report = cleaned
        assert set(report.removed_tickers) == {"EA", "CICC"}
        assert "EA" not in set(df["ticker"].unique())

    def test_removes_zero_volume_days(self, cleaned):
        df, _ = cleaned
        assert (df["volume"] == 0).sum() == 0

    def test_all_sample_zero_volume_rows_belong_to_doomed_tickers(self, raw):
        """A property of this sample worth pinning down.

        Its 9 zero-volume rows are 4 from CICC and 4 from EA -- both dropped for
        short history -- plus NEE's empty-price row. EA's last four closes are
        frozen at the same value, the signature of a delisted symbol. So the
        zero-volume filter removes nothing here that another filter would not,
        which is why the flag itself is exercised synthetically below.
        """
        zero = raw[raw["volume"] == 0]
        assert len(zero) == 9
        assert set(zero["ticker"].unique()) == {"CICC", "EA", "NEE"}

    def test_output_preserves_the_canonical_contract(self, cleaned):
        df, _ = cleaned
        assert list(df.columns) == COLUMNS
        assert df.index.equals(pd.RangeIndex(len(df)))
        assert isinstance(df["ticker"].dtype, pd.CategoricalDtype)
        assert set(df["ticker"].cat.categories) == set(df["ticker"].unique())

    def test_every_surviving_price_is_positive(self, cleaned):
        df, _ = cleaned
        assert df[["open", "high", "low", "close", "adj_close"]].gt(0).all().all()

    def test_ordering_survives_cleaning(self, cleaned):
        df, _ = cleaned
        for ticker, group in df.groupby("ticker", observed=True):
            assert group["date"].is_monotonic_increasing, ticker


class TestReportAccounting:
    def test_totals_reconcile_with_the_steps(self, cleaned):
        """The report must add up: no row vanishes unaccounted for."""
        df, report = cleaned
        assert report.rows_out == len(df)
        assert sum(s.rows_removed for s in report.steps) == report.rows_removed

    def test_every_filter_is_listed_even_when_it_removes_nothing(self, cleaned):
        _, report = cleaned
        names = [s.name for s in report.steps]
        assert names == [
            "duplicates", "missing prices", "non-positive prices",
            "incoherent OHLC", "zero volume", "short history",
        ]

    def test_report_renders(self, cleaned):
        _, report = cleaned
        text = str(report)
        assert "Cleaning report" in text
        assert "short history" in text
        assert "CICC" in text


class TestSyntheticDefects:
    """Defects the sample does not contain, so the filters are still proven."""

    def test_duplicate_ticker_date_is_removed(self):
        rows = series("AAA", 800)
        rows.append(dict(rows[5]))  # exact duplicate of one day
        df, report = clean_prices(make_frame(rows), min_history_days=10)
        assert next(s for s in report.steps if s.name == "duplicates").rows_removed == 1
        assert not df.duplicated(["ticker", "date"]).any()

    def test_non_positive_price_is_removed(self):
        rows = series("AAA", 800)
        rows[10] = {**rows[10], "low": 0.0, "open": 0.0, "close": 0.0, "adj_close": 0.0, "high": 0.0}
        df, report = clean_prices(make_frame(rows), min_history_days=10)
        assert next(s for s in report.steps if s.name == "non-positive prices").rows_removed == 1

    def test_incoherent_ohlc_is_removed(self):
        rows = series("AAA", 800)
        rows[20] = {**rows[20], "high": 5.0}  # high below low and close
        df, report = clean_prices(make_frame(rows), min_history_days=10)
        assert next(s for s in report.steps if s.name == "incoherent OHLC").rows_removed == 1

    def test_short_history_filter_runs_after_row_filters(self):
        """A ticker pushed under the threshold by row filters must be dropped.

        This is the ordering guarantee: 'GOOD' has 10 clean rows; 'BAD' has 10
        rows of which 3 are unusable, leaving 7. With a threshold of 8, BAD must
        go -- which only happens if history is counted *after* the row filters.
        """
        rows = series("GOOD", 10)
        bad = series("BAD", 10)
        for i in (0, 1, 2):
            bad[i] = {**bad[i], "volume": 0}
        df, report = clean_prices(make_frame(rows + bad), min_history_days=8)
        assert report.removed_tickers == ["BAD"]
        assert set(df["ticker"].unique()) == {"GOOD"}

    def test_zero_volume_rows_are_dropped_by_default(self):
        rows = series("AAA", 800)
        for i in (3, 4, 5):
            rows[i] = {**rows[i], "volume": 0}
        df, report = clean_prices(make_frame(rows), min_history_days=10)
        assert next(s for s in report.steps if s.name == "zero volume").rows_removed == 3
        assert (df["volume"] == 0).sum() == 0

    def test_zero_volume_rows_are_kept_when_the_flag_is_off(self):
        rows = series("AAA", 800)
        for i in (3, 4, 5):
            rows[i] = {**rows[i], "volume": 0}
        df, report = clean_prices(
            make_frame(rows), min_history_days=10, drop_zero_volume=False
        )
        assert (df["volume"] == 0).sum() == 3
        assert not any(s.name == "zero volume" for s in report.steps)

    def test_rejects_a_non_canonical_frame(self):
        with pytest.raises(ValueError, match="missing columns"):
            clean_prices(pd.DataFrame({"date": [], "ticker": []}))

    def test_clean_input_is_returned_unchanged(self):
        df_in = make_frame(series("AAA", 800))
        df_out, report = clean_prices(df_in, min_history_days=10)
        assert report.rows_removed == 0
        assert len(df_out) == len(df_in)
