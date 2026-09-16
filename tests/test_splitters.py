"""Tests for `src.evaluation.splitters`.

The important test in this file is `test_no_training_target_reaches_the_test_period`.
Everything else checks the mechanics; that one checks the guarantee the whole
evaluation protocol rests on, and it is paired with a test proving it fails when
the embargo is removed. A guard that cannot fail is not a guard.
"""

import numpy as np
import pandas as pd
import pytest

from src import config
from src.evaluation.splitters import Fold, WalkForwardSplitter


def panel(
    tickers: tuple[str, ...] = ("AAA", "BBB", "CCC"),
    start: str = "2000-01-03",
    years: int = 12,
) -> pd.DataFrame:
    """A date/ticker panel on business days, shaped like the real dataset.

    Each ticker starts a few days later than the previous one, so the tickers
    do not share a calendar -- the property that makes a calendar-day embargo
    wrong and a per-ticker row embargo right.
    """
    frames = []
    for offset, ticker in enumerate(tickers):
        dates = pd.bdate_range(start, periods=252 * years)[offset * 3:]
        frames.append(pd.DataFrame({"date": dates, "ticker": ticker}))
    df = pd.concat(frames, ignore_index=True)
    df["ticker"] = df["ticker"].astype("category")
    return df.sort_values(["ticker", "date"], kind="stable").reset_index(drop=True)


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    return panel()


@pytest.fixture(scope="module")
def splitter() -> WalkForwardSplitter:
    return WalkForwardSplitter(
        horizon=21, first_test_year=2005, last_test_year=2011,
        min_train_rows=100, min_test_rows=100,
    )


class TestNoLookahead:
    def test_training_rows_all_precede_the_test_period(self, df, splitter):
        for fold in splitter.split(df):
            assert df.loc[fold.train_idx, "date"].max() < fold.test_start

    def test_no_training_target_reaches_the_test_period(self, df, splitter):
        """The guarantee the evaluation protocol rests on.

        For every training row, the date `horizon` rows ahead *for its own
        ticker* must still fall before the test period. Otherwise that row's
        target was partly computed from days the model is about to be scored on.
        """
        target_end = df.groupby("ticker", observed=True)["date"].shift(-splitter.horizon)

        for fold in splitter.split(df):
            ends = target_end.iloc[fold.train_idx].dropna()
            assert (ends < fold.test_start).all(), (
                f"fold {fold.year}: {(ends >= fold.test_start).sum()} training "
                f"targets overlap the test period"
            )

    def test_the_guarantee_fails_without_an_embargo(self, df):
        """Proves the test above can fail. Without the gap, the last rows of
        training reach straight into the test year."""
        naive = WalkForwardSplitter(
            horizon=0, first_test_year=2005, last_test_year=2011,
            min_train_rows=100, min_test_rows=100,
        )
        target_end = df.groupby("ticker", observed=True)["date"].shift(-21)

        violations = 0
        for fold in naive.split(df):
            ends = target_end.iloc[fold.train_idx].dropna()
            violations += int((ends >= fold.test_start).sum())
        assert violations > 0

    def test_embargo_removes_horizon_rows_per_ticker(self, df, splitter):
        n_tickers = df["ticker"].nunique()
        for fold in splitter.split(df):
            assert fold.n_embargoed == splitter.horizon * n_tickers

    def test_a_zero_horizon_embargoes_nothing(self, df):
        splitter = WalkForwardSplitter(
            horizon=0, first_test_year=2005, last_test_year=2006,
            min_train_rows=100, min_test_rows=100,
        )
        assert all(fold.n_embargoed == 0 for fold in splitter.split(df))


class TestFoldStructure:
    def test_produces_one_fold_per_year_in_range(self, df, splitter):
        years = [fold.year for fold in splitter.split(df)]
        assert years == list(range(2005, 2012))

    def test_test_folds_are_disjoint(self, df, splitter):
        seen: set[int] = set()
        for fold in splitter.split(df):
            assert not seen & set(fold.test_idx.tolist())
            seen |= set(fold.test_idx.tolist())

    def test_test_folds_stay_inside_their_year(self, df, splitter):
        for fold in splitter.split(df):
            dates = df.loc[fold.test_idx, "date"]
            assert (dates.dt.year == fold.year).all()

    def test_train_and_test_never_overlap(self, df, splitter):
        for fold in splitter.split(df):
            assert not set(fold.train_idx.tolist()) & set(fold.test_idx.tolist())

    def test_indices_are_positional_and_in_range(self, df, splitter):
        for fold in splitter.split(df):
            for idx in (fold.train_idx, fold.test_idx):
                assert idx.dtype.kind in "iu"
                assert idx.min() >= 0 and idx.max() < len(df)

    def test_counts_match_the_index_arrays(self, df, splitter):
        for fold in splitter.split(df):
            assert fold.n_train == len(fold.train_idx)
            assert fold.n_test == len(fold.test_idx)


class TestWindowGrowth:
    def test_the_expanding_window_only_grows(self, df, splitter):
        sizes = [fold.n_train for fold in splitter.split(df)]
        assert sizes == sorted(sizes)
        assert sizes[-1] > sizes[0]

    def test_the_expanding_window_always_starts_at_the_beginning(self, df, splitter):
        starts = {fold.train_start for fold in splitter.split(df)}
        assert len(starts) == 1

    def test_a_rolling_window_is_bounded(self, df):
        rolling = WalkForwardSplitter(
            horizon=21, first_test_year=2007, last_test_year=2011,
            max_train_years=3, min_train_rows=100, min_test_rows=100,
        )
        for fold in rolling.split(df):
            span_years = (fold.train_end - fold.train_start).days / 365.25
            assert span_years <= 3.05

    def test_a_rolling_window_moves_forward(self, df):
        rolling = WalkForwardSplitter(
            horizon=21, first_test_year=2007, last_test_year=2011,
            max_train_years=3, min_train_rows=100, min_test_rows=100,
        )
        starts = [fold.train_start for fold in rolling.split(df)]
        assert starts == sorted(starts)
        assert starts[-1] > starts[0]


class TestGuards:
    def test_skips_folds_that_would_be_too_small(self, df):
        splitter = WalkForwardSplitter(
            horizon=21, first_test_year=2000, last_test_year=2011,
            min_train_rows=5_000, min_test_rows=100,
        )
        years = [fold.year for fold in splitter.split(df)]
        assert 2000 not in years  # no history yet
        assert 2011 in years

    def test_rejects_a_reversed_year_range(self):
        with pytest.raises(ValueError, match="after"):
            WalkForwardSplitter(first_test_year=2020, last_test_year=2010)

    def test_rejects_a_negative_horizon(self):
        with pytest.raises(ValueError, match="non-negative"):
            WalkForwardSplitter(horizon=-1)

    def test_rejects_a_zero_rolling_window(self):
        with pytest.raises(ValueError, match="at least 1"):
            WalkForwardSplitter(max_train_years=0)

    def test_requires_date_and_ticker_columns(self, splitter):
        with pytest.raises(ValueError, match="'ticker'"):
            list(splitter.split(pd.DataFrame({"date": pd.to_datetime([])})))


class TestSummary:
    def test_summary_has_one_row_per_fold(self, df, splitter):
        table = splitter.summary(df)
        assert len(table) == len(list(splitter.split(df)))
        assert list(table.columns) == [
            "year", "train_start", "train_end", "test_start", "test_end",
            "n_train", "n_test", "n_embargoed",
        ]

    def test_summary_is_reproducible(self, df, splitter):
        pd.testing.assert_frame_equal(splitter.summary(df), splitter.summary(df))


class TestDefaults:
    def test_defaults_come_from_config(self):
        splitter = WalkForwardSplitter()
        assert splitter.horizon == config.HORIZON
        assert splitter.first_test_year == config.FIRST_TEST_YEAR
        assert splitter.last_test_year == config.LAST_TEST_YEAR
        assert splitter.max_train_years is None

    def test_embargo_tracks_the_horizon(self):
        """EMBARGO is defined as HORIZON in config; the splitter must not
        introduce a second, independent notion of the gap."""
        assert WalkForwardSplitter().horizon == config.EMBARGO
