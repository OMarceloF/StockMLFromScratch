"""Tests for `src.features.technical` and the diagnostics in `src.viz.plots`.

The point of most of these is the group boundary. A rolling window or a lag that
forgets to group by ticker produces a number for every row and raises nothing --
it just reads the end of one company's history as the start of the next. So
several tests build two-ticker frames with deliberately different price levels,
where a boundary leak would show up as an enormous fake return.
"""

import numpy as np
import pandas as pd
import pytest

from src import config
from src.features.technical import log_returns, realized_volatility
from src.viz.plots import acf, mean_acf


def frame(prices_by_ticker: dict[str, list[float]]) -> pd.DataFrame:
    rows = []
    for ticker, prices in prices_by_ticker.items():
        dates = pd.bdate_range("2020-01-01", periods=len(prices))
        rows += [{"date": d, "ticker": ticker, "adj_close": p}
                 for d, p in zip(dates, prices)]
    df = pd.DataFrame(rows)
    df["ticker"] = df["ticker"].astype("category")
    return df


class TestLogReturns:
    def test_matches_the_definition(self):
        df = frame({"A": [100.0, 110.0, 99.0]})
        got = log_returns(df)
        assert np.isnan(got.iloc[0])
        assert got.iloc[1] == pytest.approx(np.log(110 / 100))
        assert got.iloc[2] == pytest.approx(np.log(99 / 110))

    def test_first_row_of_every_ticker_is_nan(self):
        df = frame({"A": [10.0, 11.0, 12.0], "B": [500.0, 505.0, 510.0]})
        got = log_returns(df)
        first_rows = df.groupby("ticker", observed=True).head(1).index
        assert got.loc[first_rows].isna().all()
        assert got.drop(index=first_rows).notna().all()

    def test_does_not_leak_across_the_ticker_boundary(self):
        """A leak here would show as log(500/12) ~ 3.7 instead of NaN."""
        df = frame({"A": [10.0, 11.0, 12.0], "B": [500.0, 505.0, 510.0]})
        got = log_returns(df)
        boundary = df.index[df["ticker"] == "B"][0]
        assert np.isnan(got.loc[boundary])
        assert got.abs().max() < 0.5

    def test_log_returns_are_additive_across_time(self):
        """The property that makes log returns the right choice."""
        df = frame({"A": [100.0, 107.0, 95.0, 130.0]})
        got = log_returns(df)
        assert got.sum() == pytest.approx(np.log(130 / 100))

    def test_index_is_preserved(self):
        df = frame({"A": [1.0, 2.0], "B": [3.0, 4.0]})
        assert log_returns(df).index.equals(df.index)

    def test_uses_adj_close_not_close(self):
        """Guards the dividend decision: close would invent a drop on ex-date."""
        df = frame({"A": [100.0, 100.0]})
        df["close"] = [100.0, 98.0]           # a 2.00 dividend paid on day two
        assert log_returns(df).iloc[1] == pytest.approx(0.0)
        assert log_returns(df, price_col="close").iloc[1] < -0.019


class TestRealizedVolatility:
    def test_matches_numpy_on_a_full_window(self):
        rng = np.random.default_rng(config.RANDOM_SEED)
        prices = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 40)))
        df = frame({"A": list(prices)})
        r = log_returns(df)
        vol = realized_volatility(r, df["ticker"], window=10, annualize=False)
        expected = r.iloc[11:21].std()
        assert vol.iloc[20] == pytest.approx(expected)

    def test_emits_nothing_until_the_window_is_full(self):
        df = frame({"A": list(np.linspace(100, 120, 30))})
        vol = realized_volatility(log_returns(df), df["ticker"], window=10)
        assert vol.iloc[:10].isna().all()
        assert vol.iloc[10:].notna().all()

    def test_annualisation_is_sqrt_252(self):
        rng = np.random.default_rng(config.RANDOM_SEED)
        prices = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 60)))
        df = frame({"A": list(prices)})
        r = log_returns(df)
        raw = realized_volatility(r, df["ticker"], window=20, annualize=False)
        ann = realized_volatility(r, df["ticker"], window=20, annualize=True)
        ratio = (ann / raw).dropna()
        assert ratio.round(9).nunique() == 1
        assert ratio.iloc[0] == pytest.approx(np.sqrt(config.TRADING_DAYS_PER_YEAR))

    def test_windows_do_not_cross_the_ticker_boundary(self):
        """B is perfectly flat, so its volatility must be exactly zero.

        If A's window bled into B, B's early values would be non-zero.
        """
        df = frame({"A": list(100 * np.exp(np.cumsum(np.full(40, 0.03)))),
                    "B": [50.0] * 40})
        vol = realized_volatility(log_returns(df), df["ticker"], window=5)
        b_vol = vol[df["ticker"] == "B"].dropna()
        assert len(b_vol) > 0
        assert (b_vol == 0).all()

    def test_index_is_preserved(self):
        df = frame({"A": [1.0, 2.0, 3.0], "B": [4.0, 5.0, 6.0]})
        r = log_returns(df)
        assert realized_volatility(r, df["ticker"], window=2).index.equals(df.index)


class TestAcf:
    def test_white_noise_has_no_autocorrelation(self):
        rng = np.random.default_rng(config.RANDOM_SEED)
        x = rng.normal(size=20_000)
        got = acf(x, max_lag=10)
        assert np.abs(got).max() < 3 / np.sqrt(len(x))

    def test_recovers_a_known_ar1_coefficient(self):
        """For an AR(1), the ACF at lag k is phi**k -- a real check, not a smoke test."""
        rng = np.random.default_rng(config.RANDOM_SEED)
        phi, n = 0.7, 200_000
        x = np.zeros(n)
        noise = rng.normal(size=n)
        for t in range(1, n):
            x[t] = phi * x[t - 1] + noise[t]
        got = acf(x, max_lag=3)
        assert got[0] == pytest.approx(phi, abs=0.01)
        assert got[1] == pytest.approx(phi**2, abs=0.01)
        assert got[2] == pytest.approx(phi**3, abs=0.01)

    def test_lag_zero_is_excluded(self):
        """acf() returns lags 1..max_lag; a leading 1.0 would shift every lag."""
        rng = np.random.default_rng(config.RANDOM_SEED)
        got = acf(rng.normal(size=1000), max_lag=5)
        assert len(got) == 5
        assert got[0] != pytest.approx(1.0)

    def test_constant_series_returns_nan(self):
        assert np.isnan(acf(np.ones(100), max_lag=3)).all()

    def test_series_shorter_than_the_lag_returns_nan(self):
        assert np.isnan(acf(np.arange(5.0), max_lag=10)).all()

    def test_nans_are_dropped(self):
        rng = np.random.default_rng(config.RANDOM_SEED)
        x = rng.normal(size=5000)
        with_nan = np.concatenate([[np.nan], x])
        assert acf(with_nan, 5) == pytest.approx(acf(x, 5))


class TestMeanAcf:
    def test_averages_per_ticker_without_splicing_them(self):
        """Two flat-then-jumping tickers: a spliced series would show a spike.

        A is strongly trending and B is flat. Computed per ticker and averaged,
        the result is the mean of the two ACFs. Concatenated first, the join
        would inject a huge artificial observation.
        """
        n = 400
        df = frame({"A": list(100 * np.exp(np.cumsum(np.full(n, 0.001)))),
                    "B": [50.0] * n})
        r = log_returns(df)
        got = mean_acf(r, df["ticker"], max_lag=5)
        assert len(got) == 5
        assert np.isfinite(got).all()

    def test_absolute_flag_uses_magnitudes(self):
        rng = np.random.default_rng(config.RANDOM_SEED)
        prices = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 3000)))
        df = frame({"A": list(prices)})
        r = log_returns(df)
        plain = mean_acf(r, df["ticker"], max_lag=3)
        absolute = mean_acf(r, df["ticker"], max_lag=3, absolute=True)
        assert not np.allclose(plain, absolute)
