"""Walk-forward splits over time, with an embargo between train and test.

A random train/test split is meaningless on this data. Rows carry dates, the
relationships being modelled drift, and shuffling lets the model fit on 2020 and
be scored on 1995 -- which no practitioner could ever have done. Measured on
this dataset, the same model reports R^2 0.5892 fitted and scored on everything
against 0.5511 walk-forward, and the single in-sample number hides a spread from
0.21 to 0.75 across individual years.

The protocol here is the one a practitioner could actually have followed:

    fold 1:  train 1990..2004  |embargo|  test 2005
    fold 2:  train 1990..2005  |embargo|  test 2006
    ...
    fold 22: train 1990..2025  |embargo|  test 2026

The training window expands rather than slides, because that is what someone
standing in 2015 would have had: every year of history up to that point.
`max_train_years` switches to a rolling window for the robustness check.

The embargo
-----------
Cutting train and test flush at the new year would leak. A target dated `t`
covers returns from `t+1` through `t+horizon`, so the final `horizon` rows of
any training fold already encode the beginning of the test period. The model
would be scored, in part, on days it had been trained on.

The gap is measured in **trading rows per ticker**, not calendar days. The
tickers do not share a calendar, so "21 days back" means a different cut for
each one; dropping each ticker's last `horizon` eligible rows is exact for that
ticker. Where the dataset has interior gaps this errs slightly on the side of
removing too much, which is the right direction for a safety margin.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src import config


@dataclass(frozen=True)
class Fold:
    """One train/test pair, plus the context needed to report it."""

    year: int
    train_idx: np.ndarray
    test_idx: np.ndarray
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    n_embargoed: int

    @property
    def n_train(self) -> int:
        return len(self.train_idx)

    @property
    def n_test(self) -> int:
        return len(self.test_idx)


class WalkForwardSplitter:
    """Expanding-window annual splits with a per-ticker embargo.

    Parameters
    ----------
    horizon
        Forecast horizon in trading days. Sets the embargo width, because the
        embargo exists precisely to cover the target's reach.
    first_test_year, last_test_year
        Inclusive range of years used as test folds.
    max_train_years
        ``None`` (default) expands the training window from the start of the
        data. An integer switches to a rolling window of that many years, used
        to check whether conclusions depend on keeping the distant past.
    min_train_rows, min_test_rows
        Folds smaller than these are skipped rather than silently producing a
        metric computed on a handful of observations.
    """

    def __init__(
        self,
        *,
        horizon: int = config.HORIZON,
        first_test_year: int = config.FIRST_TEST_YEAR,
        last_test_year: int = config.LAST_TEST_YEAR,
        max_train_years: int | None = None,
        min_train_rows: int = 10_000,
        min_test_rows: int = 1_000,
    ):
        if horizon < 0:
            raise ValueError("horizon must be non-negative")
        if first_test_year > last_test_year:
            raise ValueError(
                f"first_test_year ({first_test_year}) is after "
                f"last_test_year ({last_test_year})"
            )
        if max_train_years is not None and max_train_years < 1:
            raise ValueError("max_train_years must be at least 1")

        self.horizon = horizon
        self.first_test_year = first_test_year
        self.last_test_year = last_test_year
        self.max_train_years = max_train_years
        self.min_train_rows = min_train_rows
        self.min_test_rows = min_test_rows

    def split(self, df: pd.DataFrame) -> Iterator[Fold]:
        """Yield one `Fold` per test year.

        Parameters
        ----------
        df
            Frame with `date` and `ticker` columns, sorted by (ticker, date) --
            the shape `src.features.pipeline.build_dataset` produces.
        """
        for column in ("date", "ticker"):
            if column not in df.columns:
                raise ValueError(f"expected a '{column}' column")

        dates = df["date"]
        tickers = df["ticker"]
        positions = np.arange(len(df))

        for year in range(self.first_test_year, self.last_test_year + 1):
            test_start = pd.Timestamp(f"{year}-01-01")
            test_stop = pd.Timestamp(f"{year + 1}-01-01")

            in_test = (dates >= test_start) & (dates < test_stop)
            eligible = dates < test_start

            if self.max_train_years is not None:
                window_start = test_start - pd.DateOffset(years=self.max_train_years)
                eligible = eligible & (dates >= window_start)

            in_train = self._apply_embargo(eligible, tickers)

            n_train = int(in_train.sum())
            n_test = int(in_test.sum())
            if n_train < self.min_train_rows or n_test < self.min_test_rows:
                continue

            yield Fold(
                year=year,
                train_idx=positions[in_train.to_numpy()],
                test_idx=positions[in_test.to_numpy()],
                train_start=dates[in_train].min(),
                train_end=dates[in_train].max(),
                test_start=dates[in_test].min(),
                test_end=dates[in_test].max(),
                n_embargoed=int(eligible.sum()) - n_train,
            )

    def _apply_embargo(self, eligible: pd.Series, tickers: pd.Series) -> pd.Series:
        """Drop each ticker's final `horizon` eligible rows.

        `cumcount(ascending=False)` numbers rows backwards from the end of each
        ticker's eligible block: 0 for the last, 1 for the one before it, and so
        on. Keeping only positions at or beyond `horizon` removes exactly the
        rows whose target window reaches into the test period.
        """
        if self.horizon == 0:
            return eligible

        eligible_tickers = tickers[eligible]
        distance_from_end = (
            eligible_tickers.groupby(eligible_tickers, observed=True)
            .cumcount(ascending=False)
        )
        keep = distance_from_end >= self.horizon
        return eligible & keep.reindex(eligible.index, fill_value=False)

    def summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """A table of the folds this splitter produces, for reporting."""
        return pd.DataFrame([
            {
                "year": fold.year,
                "train_start": fold.train_start.date(),
                "train_end": fold.train_end.date(),
                "test_start": fold.test_start.date(),
                "test_end": fold.test_end.date(),
                "n_train": fold.n_train,
                "n_test": fold.n_test,
                "n_embargoed": fold.n_embargoed,
            }
            for fold in self.split(df)
        ])
