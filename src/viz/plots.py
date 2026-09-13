"""Styled matplotlib helpers, so every figure in the README is one call away.

Figure code lives here rather than inline in notebooks for two reasons: the
charts stay consistent with one another, and a plot that appears in the README
can be regenerated without hunting for the cell that made it.

Palette
-------
Two categorical slots (blue `#2a78d6`, orange `#eb6834`) carry every comparison
in this project. The pair was checked rather than eyeballed -- worst-case colour
distance under simulated protanopia is dE 24.7 and under tritanopia 32.7, both
far above the dE 8 threshold, and both clear 3:1 contrast against the chart
surface. Every chart with two series also carries a legend, so identity is never
colour alone.

Figures are rendered on an explicit light surface rather than a transparent one:
a transparent PNG inherits whatever background the viewer's README theme
supplies, which turns dark axis text invisible in dark mode.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------

SERIES_1 = "#2a78d6"  # blue
SERIES_2 = "#eb6834"  # orange
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
AXIS = "#c3c2b7"


def set_style() -> None:
    """Apply the project's chart style. Call once per notebook."""
    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "semibold",
        "axes.titlecolor": INK_PRIMARY,
        "axes.titlepad": 12,
        "axes.labelsize": 10,
        "axes.labelcolor": INK_SECONDARY,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRIDLINE,
        "grid.linewidth": 0.8,
        "grid.linestyle": "-",          # never dashed: dashing reads as "threshold"
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "lines.linewidth": 1.8,          # thin marks
        "figure.dpi": 110,
        "savefig.dpi": 160,
        "savefig.bbox": "tight",
    })


def save_figure(fig: plt.Figure, name: str) -> Path:
    """Write a figure to `reports/figures/` and return the path."""
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = config.FIGURES_DIR / f"{name}.png"
    fig.savefig(path)
    return path


def _annotate(ax: plt.Axes, text: str, *, loc: str = "upper right") -> None:
    """A small block of supporting numbers, in ink rather than series colour."""
    x, ha = (0.98, "right") if "right" in loc else (0.02, "left")
    y, va = (0.97, "top") if "upper" in loc else (0.03, "bottom")
    ax.text(
        x, y, text, transform=ax.transAxes, ha=ha, va=va,
        fontsize=9, color=INK_SECONDARY, linespacing=1.5,
    )


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def acf(x: np.ndarray, max_lag: int) -> np.ndarray:
    """Sample autocorrelation of `x` for lags 1..max_lag.

    Implemented directly rather than pulled from statsmodels: it is six lines,
    it keeps the dependency list short, and the point of the figure it feeds is
    to make the reader trust the number.

    NaNs are dropped first, which is safe here because the input is one ticker's
    contiguous return series whose only NaN is the unavoidable first row.
    """
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n <= max_lag + 1:
        return np.full(max_lag, np.nan)

    x = x - x.mean()
    denominator = np.dot(x, x)
    if denominator == 0:
        return np.full(max_lag, np.nan)

    return np.array([np.dot(x[lag:], x[:-lag]) / denominator for lag in range(1, max_lag + 1)])


def mean_acf(
    returns: pd.Series,
    groups: pd.Series,
    max_lag: int,
    *,
    absolute: bool = False,
) -> np.ndarray:
    """Average the per-ticker ACF across the panel.

    Pooling the raw series and taking one ACF would splice each ticker's last
    observation onto the next ticker's first. Computing per ticker and then
    averaging keeps every lag honest.
    """
    values = returns.abs() if absolute else returns
    per_ticker = [
        acf(group.to_numpy(), max_lag)
        for _, group in values.groupby(groups, observed=True)
    ]
    return np.nanmean(np.vstack(per_ticker), axis=0)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_universe_coverage(df: pd.DataFrame) -> plt.Figure:
    """How many tickers are observable over time, and how long each history is."""
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(11, 3.8))

    by_year = df.groupby(df["date"].dt.year, observed=True)["ticker"].nunique()
    ax_left.plot(by_year.index, by_year.to_numpy(), color=SERIES_1)
    ax_left.fill_between(by_year.index, by_year.to_numpy(), color=SERIES_1, alpha=0.12)
    ax_left.set_title("Tickers observable each year")
    ax_left.set_xlabel("year")
    ax_left.set_ylabel("tickers")
    ax_left.set_ylim(0, by_year.max() * 1.12)
    last_year, last_value = by_year.index[-1], by_year.iloc[-1]
    ax_left.annotate(
        f"{last_value}", xy=(last_year, last_value), xytext=(-4, 8),
        textcoords="offset points", ha="right", fontsize=9, color=INK_SECONDARY,
    )

    counts = df.groupby("ticker", observed=True).size()
    years = counts / config.TRADING_DAYS_PER_YEAR
    ax_right.hist(years, bins=30, color=SERIES_1, alpha=0.25,
                  edgecolor=SERIES_1, linewidth=1.2)
    ax_right.set_title("Length of each ticker's history")
    ax_right.set_xlabel("years of data")
    ax_right.set_ylabel("tickers")
    # Upper left: the distribution's mass sits at the right-hand edge, where
    # most tickers have the full 36-year history, so an upper-right annotation
    # lands on top of the tallest bar.
    _annotate(
        ax_right,
        f"median  {years.median():.1f} y\n"
        f"min     {years.min():.1f} y\n"
        f"max     {years.max():.1f} y",
        loc="upper left",
    )

    fig.tight_layout()
    return fig


def plot_return_distribution(returns: pd.Series) -> plt.Figure:
    """Daily log returns against the normal distribution with the same moments.

    A log-scaled y-axis is essential: on a linear axis the two curves look
    identical, because the disagreement lives entirely in tails that are a
    thousand times shorter than the peak.
    """
    r = returns.dropna().to_numpy()
    mu, sigma = r.mean(), r.std()

    fig, ax = plt.subplots(figsize=(8, 4.4))
    limit = 0.15
    # `range=` discards observations outside the window. Clipping them into the
    # edge bins instead would raise two spikes at +/-0.15 that are an artefact
    # of the plotting, not a feature of the data.
    ax.hist(r, bins=240, range=(-limit, limit), density=True,
            color=SERIES_1, alpha=0.25, edgecolor=SERIES_1, linewidth=0.9,
            label="observed returns")

    grid = np.linspace(-limit, limit, 600)
    normal = np.exp(-0.5 * ((grid - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
    ax.plot(grid, normal, color=SERIES_2, label="normal, same mean and sd")
    ax.set_xlim(-limit, limit)

    ax.set_yscale("log")
    ax.set_title("Daily log returns have far fatter tails than a normal distribution")
    ax.set_xlabel("daily log return")
    ax.set_ylabel("density (log scale)")
    ax.legend(loc="upper right")

    excess_kurtosis = ((r - mu) ** 4).mean() / sigma**4 - 3
    beyond_5 = int((np.abs(r - mu) > 5 * sigma).sum())
    expected_5 = len(r) * 5.733e-7  # P(|Z| > 5) under a normal
    off_chart = int((np.abs(r) > limit).sum())
    _annotate(
        ax,
        f"excess kurtosis   {excess_kurtosis:,.1f}\n"
        f"days beyond 5 sd  {beyond_5:,}\n"
        f"normal predicts   {expected_5:,.1f}\n"
        f"beyond this axis  {off_chart:,}",
        loc="upper left",
    )

    fig.tight_layout()
    return fig


def plot_volatility_clustering(
    dates: pd.Series, market_returns: np.ndarray, market_vol: np.ndarray
) -> plt.Figure:
    """Turbulence arrives in bursts -- the empirical basis for Experiment 1."""
    fig, (ax_top, ax_bottom) = plt.subplots(
        2, 1, figsize=(11, 5.6), sharex=True, height_ratios=[1, 1]
    )

    ax_top.plot(dates, market_returns, color=SERIES_1, linewidth=0.5)
    ax_top.set_title("Market daily return, and its 21-day realised volatility")
    ax_top.set_ylabel("daily log return")
    ax_top.axhline(0, color=AXIS, linewidth=0.8)

    ax_bottom.plot(dates, market_vol * 100, color=SERIES_2)
    ax_bottom.fill_between(dates, market_vol * 100, color=SERIES_2, alpha=0.12)
    ax_bottom.set_ylabel("annualised volatility (%)")
    ax_bottom.set_xlabel("date")

    fig.tight_layout()
    return fig


def plot_acf_comparison(
    lags: np.ndarray,
    acf_returns: np.ndarray,
    acf_abs_returns: np.ndarray,
    n_obs: int,
) -> plt.Figure:
    """The chart the whole project rests on.

    If returns were predictable from their own past, the blue line would sit
    away from zero. It does not. If volatility were unpredictable, the orange
    line would do the same. It does not either. That asymmetry is the reason
    Experiment 1 targets volatility and Experiment 2 is filed as a negative
    control rather than as the headline.
    """
    fig, ax = plt.subplots(figsize=(9, 4.4))

    band = 1.96 / np.sqrt(n_obs)
    ax.fill_between([lags[0], lags[-1]], -band, band, color=GRIDLINE, alpha=0.9,
                    label="95% band under no autocorrelation")
    ax.axhline(0, color=AXIS, linewidth=0.8)

    ax.plot(lags, acf_returns, color=SERIES_1, marker="o", markersize=3,
            label="return  $r_t$")
    ax.plot(lags, acf_abs_returns, color=SERIES_2, marker="o", markersize=3,
            label="absolute return  $|r_t|$")

    ax.annotate("$|r_t|$", xy=(lags[-1], acf_abs_returns[-1]), xytext=(6, 0),
                textcoords="offset points", va="center", fontsize=9, color=INK_SECONDARY)
    ax.annotate("$r_t$", xy=(lags[-1], acf_returns[-1]), xytext=(6, 0),
                textcoords="offset points", va="center", fontsize=9, color=INK_SECONDARY)

    ax.set_title("Returns have no memory; their magnitude has a great deal")
    ax.set_xlabel("lag (trading days)")
    ax.set_ylabel("mean autocorrelation across tickers")
    ax.set_xlim(lags[0] - 0.5, lags[-1] + 3)
    ax.legend(loc="upper right")

    fig.tight_layout()
    return fig
