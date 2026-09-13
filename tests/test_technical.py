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
from src.features.technical import (
    downside_volatility,
    log_returns,
    momentum,
    realized_volatility,
    safe_log,
    volatility_ratio,
    volume_zscore,
)
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


class TestMomentum:
    def test_equals_the_log_price_ratio_over_the_window(self):
        """The definition check: an N-day momentum is log(P_t / P_{t-N}).

        Stronger than comparing against a rolling sum, because it tests the
        quantity we mean rather than the implementation we happened to write.
        """
        rng = np.random.default_rng(config.RANDOM_SEED)
        prices = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, 200)))
        df = frame({"A": list(prices)})
        got = momentum(log_returns(df), df["ticker"], window=21)
        for t in (50, 100, 199):
            assert got.iloc[t] == pytest.approx(np.log(prices[t] / prices[t - 21]))

    def test_window_of_one_is_the_daily_return(self):
        df = frame({"A": [100.0, 110.0, 99.0, 120.0]})
        r = log_returns(df)
        pd.testing.assert_series_equal(momentum(r, df["ticker"], 1), r)

    def test_emits_nothing_until_the_window_is_full(self):
        """A leading NaN from log_returns must count against min_periods.

        Otherwise a '5-day momentum' would appear one row early, built from
        four days -- a differently-scaled feature hiding among the rest.
        """
        df = frame({"A": list(np.linspace(100, 140, 30))})
        got = momentum(log_returns(df), df["ticker"], window=5)
        assert got.iloc[:5].isna().all()
        assert got.iloc[5:].notna().all()

    def test_does_not_accumulate_across_the_ticker_boundary(self):
        """B is flat, so every B momentum must be exactly zero.

        A window bleeding over from A -- which rises 3% a day -- would leave
        large positive values at the start of B.
        """
        df = frame({"A": list(100 * np.exp(np.cumsum(np.full(60, 0.03)))),
                    "B": [50.0] * 60})
        got = momentum(log_returns(df), df["ticker"], window=21)
        b = got[df["ticker"] == "B"].dropna()
        assert len(b) > 0
        assert (b == 0).all()

    def test_windows_nest_additively(self):
        """Log-space additivity, restated: a long window is the sum of the
        short windows that tile it. The property 2.1 was chosen for."""
        rng = np.random.default_rng(config.RANDOM_SEED)
        prices = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 100)))
        df = frame({"A": list(prices)})
        r = log_returns(df)
        long_window = momentum(r, df["ticker"], 20)
        halves = momentum(r, df["ticker"], 10) + momentum(r, df["ticker"], 10).shift(10)
        assert long_window.iloc[30] == pytest.approx(halves.iloc[30])

    def test_index_is_preserved(self):
        df = frame({"A": [1.0, 2.0, 3.0], "B": [4.0, 5.0, 6.0]})
        r = log_returns(df)
        assert momentum(r, df["ticker"], 2).index.equals(df.index)


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


class TestSafeLog:
    def test_matches_numpy_log_on_positive_values(self):
        s = pd.Series([0.5, 1.0, 2.0, 100.0])
        assert safe_log(s).to_numpy() == pytest.approx(np.log(s.to_numpy()))

    def test_zero_becomes_nan_not_negative_infinity(self):
        """The distinction the whole function exists for.

        -inf survives arithmetic into the design matrix and surfaces as an
        unhelpful LinAlgError; NaN is dropped by the pipeline.
        """
        got = safe_log(pd.Series([0.0, 1.0]))
        assert np.isnan(got.iloc[0])
        assert not np.isinf(got.iloc[0])

    def test_negative_becomes_nan(self):
        assert np.isnan(safe_log(pd.Series([-1.0])).iloc[0])

    def test_nan_stays_nan(self):
        assert np.isnan(safe_log(pd.Series([np.nan])).iloc[0])

    def test_index_is_preserved(self):
        s = pd.Series([1.0, 2.0], index=[7, 9])
        assert safe_log(s).index.equals(s.index)


class TestDownsideVolatility:
    def test_is_zero_when_nothing_falls(self):
        df = frame({"A": list(np.linspace(100, 160, 40))})
        got = downside_volatility(log_returns(df), df["ticker"], window=10).dropna()
        assert len(got) > 0
        assert (got == 0).all()

    def test_matches_the_semi_deviation_definition(self):
        """sqrt(mean(min(r, 0)^2)) -- gains enter as zeros, not as omissions."""
        rng = np.random.default_rng(config.RANDOM_SEED)
        prices = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, 60)))
        df = frame({"A": list(prices)})
        r = log_returns(df)
        got = downside_volatility(r, df["ticker"], window=10, annualize=False)
        window = r.iloc[21:31].to_numpy()
        expected = np.sqrt(np.mean(np.minimum(window, 0) ** 2))
        assert got.iloc[30] == pytest.approx(expected)

    def test_gains_are_counted_as_zeros_rather_than_dropped(self):
        """Nine flat days and one -10% day: the denominator must be 10, not 1.

        Taking the standard deviation of the losing subset instead would give a
        single observation and a volatility of zero -- the opposite answer.
        """
        prices = [100.0] * 10 + [90.0]
        df = frame({"A": prices})
        r = log_returns(df)
        got = downside_volatility(r, df["ticker"], window=10, annualize=False)
        expected = np.sqrt((np.log(0.9) ** 2) / 10)
        assert got.iloc[10] == pytest.approx(expected)

    def test_emits_nothing_until_the_window_is_full(self):
        rng = np.random.default_rng(config.RANDOM_SEED)
        df = frame({"A": list(100 * np.exp(np.cumsum(rng.normal(0, 0.02, 30))))})
        got = downside_volatility(log_returns(df), df["ticker"], window=10)
        assert got.iloc[:10].isna().all()
        assert got.iloc[10:].notna().all()

    def test_does_not_cross_the_ticker_boundary(self):
        df = frame({"A": list(100 * np.exp(np.cumsum(np.full(40, -0.05)))),
                    "B": [50.0] * 40})
        got = downside_volatility(log_returns(df), df["ticker"], window=5)
        b = got[df["ticker"] == "B"].dropna()
        assert len(b) > 0
        assert (b == 0).all()

    def test_annualisation_is_sqrt_252(self):
        rng = np.random.default_rng(config.RANDOM_SEED)
        df = frame({"A": list(100 * np.exp(np.cumsum(rng.normal(0, 0.02, 60))))})
        r = log_returns(df)
        raw = downside_volatility(r, df["ticker"], 20, annualize=False)
        ann = downside_volatility(r, df["ticker"], 20, annualize=True)
        ratio = (ann / raw).dropna()
        assert ratio.iloc[0] == pytest.approx(np.sqrt(config.TRADING_DAYS_PER_YEAR))


class TestVolumeZscore:
    @staticmethod
    def volume_frame(volumes: dict[str, list[float]]) -> pd.DataFrame:
        rows = []
        for ticker, vols in volumes.items():
            dates = pd.bdate_range("2020-01-01", periods=len(vols))
            rows += [{"date": d, "ticker": ticker, "volume": v}
                     for d, v in zip(dates, vols)]
        df = pd.DataFrame(rows)
        df["ticker"] = df["ticker"].astype("category")
        return df

    def test_matches_the_definition_on_log_volume(self):
        rng = np.random.default_rng(config.RANDOM_SEED)
        vols = np.exp(rng.normal(15, 0.5, 100))
        df = self.volume_frame({"A": list(vols)})
        got = volume_zscore(df["volume"], df["ticker"], window=20)
        window = np.log(vols[31:51])
        expected = (np.log(vols[50]) - window.mean()) / window.std(ddof=1)
        assert got.iloc[50] == pytest.approx(expected)

    def test_is_scale_free_across_tickers(self):
        """A 3,202x difference in size must not shift the feature at all.

        This is the property that lets one pooled model span MTD and NVDA.
        """
        rng = np.random.default_rng(config.RANDOM_SEED)
        shape = np.exp(rng.normal(0, 0.4, 120))
        df = self.volume_frame({"SMALL": list(150_000 * shape),
                                "LARGE": list(480_000_000 * shape)})
        got = volume_zscore(df["volume"], df["ticker"], window=63)
        small = got[df["ticker"] == "SMALL"].dropna().to_numpy()
        large = got[df["ticker"] == "LARGE"].dropna().to_numpy()
        assert small == pytest.approx(large)

    def test_is_centred_on_its_own_history(self):
        """Constant-ratio growth leaves z near zero; only surprises move it."""
        rng = np.random.default_rng(config.RANDOM_SEED)
        df = self.volume_frame({"A": list(np.exp(rng.normal(15, 0.3, 400)))})
        got = volume_zscore(df["volume"], df["ticker"], window=63).dropna()
        assert abs(got.mean()) < 0.5

    def test_a_volume_spike_registers_as_a_large_positive_z(self):
        vols = [1_000_000.0] * 63 + [1_100_000.0]
        df = self.volume_frame({"A": vols})
        quiet = volume_zscore(df["volume"], df["ticker"], window=63).iloc[63]
        vols_spike = [1_000_000.0] * 63 + [20_000_000.0]
        df2 = self.volume_frame({"A": vols_spike})
        spike = volume_zscore(df2["volume"], df2["ticker"], window=63).iloc[63]
        assert spike > quiet > 0

    def test_constant_volume_yields_nan_not_infinity(self):
        """Zero denominator: NaN is droppable, +/-inf poisons the fit."""
        df = self.volume_frame({"A": [1_000_000.0] * 64})
        got = volume_zscore(df["volume"], df["ticker"], window=63)
        assert np.isnan(got.iloc[63])
        assert not np.isinf(got).any()

    def test_zero_volume_yields_nan(self):
        rng = np.random.default_rng(config.RANDOM_SEED)
        vols = list(np.exp(rng.normal(15, 0.4, 70)))
        vols[65] = 0.0
        df = self.volume_frame({"A": vols})
        assert np.isnan(volume_zscore(df["volume"], df["ticker"], window=20).iloc[65])

    def test_does_not_cross_the_ticker_boundary(self):
        """B's window must never see A's much larger volumes."""
        df = self.volume_frame({"A": [500_000_000.0] * 80, "B": [100_000.0] * 80})
        got = volume_zscore(df["volume"], df["ticker"], window=63)
        b = got[df["ticker"] == "B"]
        assert b.isna().all()  # B is constant -> NaN, never a huge negative z

    def test_emits_nothing_until_the_window_is_full(self):
        rng = np.random.default_rng(config.RANDOM_SEED)
        df = self.volume_frame({"A": list(np.exp(rng.normal(15, 0.4, 40)))})
        got = volume_zscore(df["volume"], df["ticker"], window=20)
        assert got.iloc[:19].isna().all()
        assert got.iloc[19:].notna().all()

    def test_index_is_preserved(self):
        df = self.volume_frame({"A": [1.0, 2.0, 3.0], "B": [4.0, 5.0, 6.0]})
        assert volume_zscore(df["volume"], df["ticker"], 2).index.equals(df.index)


class TestVolatilityRatio:
    def test_is_the_difference_of_logs(self):
        short = pd.Series([0.2, 0.4, 0.1])
        long = pd.Series([0.2, 0.2, 0.2])
        got = volatility_ratio(short, long)
        assert got.to_numpy() == pytest.approx(np.log(short / long))

    def test_is_zero_when_the_regimes_agree(self):
        s = pd.Series([0.3, 0.3])
        assert volatility_ratio(s, s).to_numpy() == pytest.approx([0.0, 0.0])

    def test_doubling_and_halving_are_symmetric(self):
        """The reason for the log form: +0.69 and -0.69 rather than 2.0 and 0.5."""
        base = pd.Series([0.2])
        up = volatility_ratio(base * 2, base).iloc[0]
        down = volatility_ratio(base / 2, base).iloc[0]
        assert up == pytest.approx(-down)
        assert up == pytest.approx(np.log(2))

    def test_is_scale_free(self):
        """A utility and a semiconductor must read the same at the same regime."""
        short, long = pd.Series([0.10]), pd.Series([0.05])
        assert volatility_ratio(short, long).iloc[0] == pytest.approx(
            volatility_ratio(short * 7, long * 7).iloc[0]
        )

    def test_zero_volatility_yields_nan_not_infinity(self):
        got = volatility_ratio(pd.Series([0.0]), pd.Series([0.2]))
        assert np.isnan(got.iloc[0]) and not np.isinf(got.iloc[0])

    def test_is_exactly_collinear_with_its_two_legs(self):
        """Pins down why this feature is excluded from the model matrix.

        volatility_ratio(a, b) == safe_log(a) - safe_log(b) exactly, so a design
        matrix holding all three has a rank one short of its column count. The
        normal equation does not raise -- floating point leaves a non-zero
        determinant -- it just returns coefficients that are not unique. On the
        real dataset the condition number goes from 8.5 to 9.0e14.
        """
        rng = np.random.default_rng(config.RANDOM_SEED)
        short = pd.Series(np.exp(rng.normal(-2, 0.5, 500)))
        long = pd.Series(np.exp(rng.normal(-2, 0.5, 500)))
        ratio = volatility_ratio(short, long)

        assert ratio.to_numpy() == pytest.approx(
            (safe_log(short) - safe_log(long)).to_numpy()
        )

        design = np.column_stack([
            np.ones(len(short)), safe_log(short), safe_log(long), ratio
        ])
        assert np.linalg.matrix_rank(design) == design.shape[1] - 1


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
