"""Tests for `src.features.targets`.

Two things need proving here, and only one of them is arithmetic.

The arithmetic: each target is the quantity it claims to be. Those tests build
tiny series whose answer can be written down by hand.

The other: each target looks at `t+1 .. t+horizon` and **never** at `t`. That is
not checkable by inspecting one number, so it is tested structurally -- change
the data at `t` and the target must not move; change the data after `t` and it
must. A target that quietly included its own row would pass every arithmetic
test while making the whole project's results meaningless.
"""

import numpy as np
import pandas as pd
import pytest

from src import config
from src.features.targets import (
    TARGET_COLUMNS,
    build_targets,
    forward_direction,
    forward_return,
    forward_volatility,
    next_close,
    shift_backward,
)
from src.features.technical import log_returns


def frame(prices_by_ticker: dict[str, list[float]]) -> pd.DataFrame:
    rows = []
    for ticker, prices in prices_by_ticker.items():
        dates = pd.bdate_range("2020-01-01", periods=len(prices))
        rows += [{"date": d, "ticker": ticker, "close": p, "adj_close": p}
                 for d, p in zip(dates, prices)]
    df = pd.DataFrame(rows)
    df["ticker"] = df["ticker"].astype("category")
    return df


class TestShiftBackward:
    def test_pulls_a_future_value_onto_the_current_row(self):
        df = frame({"A": [1.0, 2.0, 3.0, 4.0, 5.0]})
        got = shift_backward(df["close"], df["ticker"], 2)
        assert got.iloc[0] == 3.0
        assert got.iloc[2] == 5.0

    def test_the_last_rows_of_each_ticker_become_nan(self):
        df = frame({"A": [1.0, 2.0, 3.0], "B": [10.0, 20.0, 30.0]})
        got = shift_backward(df["close"], df["ticker"], 2)
        assert got.isna().sum() == 4  # two per ticker

    def test_does_not_splice_one_ticker_onto_another(self):
        """The bug this function exists to prevent.

        A bare shift(-1) would hand B's first price to A's last row. Across 231
        tickers that is 231 silently wrong targets and no error anywhere.
        """
        df = frame({"A": [1.0, 2.0, 3.0], "B": [100.0, 200.0, 300.0]})
        got = shift_backward(df["close"], df["ticker"], 1)
        last_a = df.index[df["ticker"] == "A"][-1]
        assert np.isnan(got.loc[last_a])
        assert 100.0 not in set(got.dropna())

    def test_index_is_preserved(self):
        df = frame({"A": [1.0, 2.0], "B": [3.0, 4.0]})
        assert shift_backward(df["close"], df["ticker"], 1).index.equals(df.index)


class TestNextClose:
    def test_is_tomorrows_close(self):
        df = frame({"A": [10.0, 11.0, 12.5]})
        got = next_close(df["close"], df["ticker"])
        assert got.iloc[0] == 11.0
        assert got.iloc[1] == 12.5
        assert np.isnan(got.iloc[2])


class TestForwardReturn:
    def test_equals_the_log_price_ratio_over_the_future_window(self):
        """forward_return at t is log(P_{t+h} / P_t) -- the definition."""
        rng = np.random.default_rng(config.RANDOM_SEED)
        prices = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, 100)))
        df = frame({"A": list(prices)})
        got = forward_return(log_returns(df), df["ticker"], horizon=21)
        for t in (0, 30, 78):
            assert got.iloc[t] == pytest.approx(np.log(prices[t + 21] / prices[t]))

    def test_uses_only_returns_after_t(self):
        """The single most important property in this file.

        Note the subtlety it encodes. `forward_return` at `t` is
        `log(P_{t+h} / P_t)`, so it legitimately reads the *price* at `t` -- that
        is the price you would transact at. What it must never read is any
        *return* at or before `t`, because those are what the features describe.

        So: perturbing a price strictly before `t` must leave the target at `t`
        untouched, while perturbing one inside `t+1 .. t+h` must move it.
        """
        rng = np.random.default_rng(config.RANDOM_SEED)
        prices = list(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 30))))
        t, horizon = 10, 5

        def target_at(series, index):
            df = frame({"A": series})
            return forward_return(log_returns(df), df["ticker"], horizon).iloc[index]

        base = target_at(prices, t)

        before = list(prices)
        before[3] *= 1.5            # inside the feature window, before t
        assert target_at(before, t) == pytest.approx(base)

        # The sum telescopes -- r_11 + ... + r_15 collapses to log(P_15 / P_10)
        # -- so an intermediate price cancels itself out entirely. Worth pinning
        # down: it means this target is indifferent to the path taken between
        # the two endpoints, which is exactly what a holding-period return is.
        midway = list(prices)
        midway[13] *= 1.5
        assert target_at(midway, t) == pytest.approx(base)

        endpoint = list(prices)
        endpoint[t + horizon] *= 1.5    # where the position is closed
        assert not np.isclose(target_at(endpoint, t), base)

        at_t = list(prices)
        at_t[t] *= 1.5                  # where the position is opened
        assert not np.isclose(target_at(at_t, t), base)

    def test_the_final_rows_have_no_target(self):
        df = frame({"A": list(np.linspace(100, 130, 40))})
        got = forward_return(log_returns(df), df["ticker"], horizon=21)
        assert got.iloc[-21:].isna().all()
        assert got.iloc[:-21].notna().all()

    def test_does_not_look_across_the_ticker_boundary(self):
        df = frame({"A": [100.0] * 30, "B": [1000.0] * 30})
        got = forward_return(log_returns(df), df["ticker"], horizon=5)
        a_rows = got[df["ticker"] == "A"]
        assert a_rows.iloc[-5:].isna().all()
        assert a_rows.dropna().to_numpy() == pytest.approx(0.0)


class TestForwardVolatility:
    def test_measures_the_window_after_t_not_before(self):
        """Calm for 40 days, then violent for 40.

        At the boundary the trailing volatility is still near zero while the
        forward volatility is already large. A target that used the trailing
        window would show the opposite.
        """
        rng = np.random.default_rng(config.RANDOM_SEED)
        calm = np.exp(np.cumsum(rng.normal(0, 0.001, 40)))
        wild = np.exp(np.cumsum(rng.normal(0, 0.05, 41)))
        prices = list(100 * calm) + list(100 * calm[-1] * wild)
        df = frame({"A": prices})
        r = log_returns(df)

        forward = forward_volatility(r, df["ticker"], horizon=21, log=False)
        # At t=18 the forward window is returns 19..39, entirely inside the calm
        # stretch. At t=19 it would already reach return 40, the first violent
        # one -- which is itself a useful reminder of how sharply this target
        # turns over at a regime boundary.
        assert forward.iloc[18] < 0.05
        assert forward.iloc[45] > 0.30      # window 46..66 is violent

    def test_matches_trailing_volatility_shifted(self):
        rng = np.random.default_rng(config.RANDOM_SEED)
        prices = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, 120)))
        df = frame({"A": list(prices)})
        r = log_returns(df)
        from src.features.technical import realized_volatility

        trailing = realized_volatility(r, df["ticker"], 21)
        forward = forward_volatility(r, df["ticker"], 21, log=False)
        assert forward.iloc[50] == pytest.approx(trailing.iloc[71])

    def test_log_is_the_default(self):
        rng = np.random.default_rng(config.RANDOM_SEED)
        df = frame({"A": list(100 * np.exp(np.cumsum(rng.normal(0, 0.02, 120))))})
        r = log_returns(df)
        logged = forward_volatility(r, df["ticker"], 21)
        raw = forward_volatility(r, df["ticker"], 21, log=False)
        assert logged.dropna().to_numpy() == pytest.approx(np.log(raw.dropna().to_numpy()))

    def test_zero_forward_volatility_yields_nan_not_negative_infinity(self):
        df = frame({"A": [100.0] * 60})
        got = forward_volatility(log_returns(df), df["ticker"], horizon=21)
        assert not np.isinf(got).any()
        assert got.dropna().empty or got.dropna().isna().all()


class TestForwardDirection:
    def test_is_one_for_a_rising_window_and_zero_for_a_falling_one(self):
        up = frame({"A": list(100 * 1.01 ** np.arange(30))})
        down = frame({"A": list(100 * 0.99 ** np.arange(30))})
        assert forward_direction(log_returns(up), up["ticker"], 5).dropna().eq(1.0).all()
        assert forward_direction(log_returns(down), down["ticker"], 5).dropna().eq(0.0).all()

    def test_stays_float_so_missing_targets_remain_missing(self):
        """An int column cannot hold NaN; casting early would relabel every
        ticker's final rows as losses."""
        df = frame({"A": list(np.linspace(100, 130, 40))})
        got = forward_direction(log_returns(df), df["ticker"], horizon=21)
        assert got.dtype == np.float64
        assert got.iloc[-21:].isna().all()

    def test_agrees_with_the_sign_of_forward_return(self):
        rng = np.random.default_rng(config.RANDOM_SEED)
        df = frame({"A": list(100 * np.exp(np.cumsum(rng.normal(0, 0.02, 300))))})
        r = log_returns(df)
        ret = forward_return(r, df["ticker"], 21)
        direction = forward_direction(r, df["ticker"], 21)
        usable = ret.notna()
        assert (direction[usable] == (ret[usable] > 0).astype(float)).all()

    def test_a_flat_window_counts_as_not_rising(self):
        df = frame({"A": [100.0] * 30})
        got = forward_direction(log_returns(df), df["ticker"], 5).dropna()
        assert (got == 0.0).all()


class TestBuildTargets:
    def test_produces_every_target_column(self):
        rng = np.random.default_rng(config.RANDOM_SEED)
        df = frame({"A": list(100 * np.exp(np.cumsum(rng.normal(0, 0.02, 120))))})
        got = build_targets(df, horizon=21)
        assert list(got.columns) == TARGET_COLUMNS
        assert got.index.equals(df.index)

    def test_every_ticker_loses_exactly_horizon_rows(self):
        rng = np.random.default_rng(config.RANDOM_SEED)
        df = frame({
            "A": list(100 * np.exp(np.cumsum(rng.normal(0, 0.02, 200)))),
            "B": list(50 * np.exp(np.cumsum(rng.normal(0, 0.02, 200)))),
        })
        got = build_targets(df, horizon=21)
        for ticker in ("A", "B"):
            rows = got.loc[df["ticker"] == ticker, "y_ret_fwd"]
            assert rows.iloc[-21:].isna().all()
            assert rows.iloc[-22] == rows.iloc[-22]  # not NaN

    def test_accepts_precomputed_returns(self):
        rng = np.random.default_rng(config.RANDOM_SEED)
        df = frame({"A": list(100 * np.exp(np.cumsum(rng.normal(0, 0.02, 120))))})
        r = log_returns(df)
        pd.testing.assert_frame_equal(
            build_targets(df, horizon=21),
            build_targets(df, horizon=21, returns=r),
        )
