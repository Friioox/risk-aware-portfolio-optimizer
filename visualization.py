"""
visualization.py
================
Professional-grade chart library for portfolio analysis.

Charts produced
---------------
1. Efficient frontier (with random portfolios for context)
2. Portfolio allocation (pie + bar)
3. Risk contribution (bar chart)
4. Drawdown chart
5. Cumulative performance
6. Rolling Sharpe ratio
7. Correlation heatmap
8. Return distribution + VaR/CVaR overlay

All functions accept a matplotlib Axes or create their own figure.
Output can be saved to file or displayed interactively.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------

PALETTE = {
    "primary": "#1f4e79",
    "accent": "#2e86c1",
    "highlight": "#f39c12",
    "danger": "#c0392b",
    "success": "#27ae60",
    "neutral": "#7f8c8d",
    "background": "#f8f9fa",
    "grid": "#dee2e6",
}

FONT = {"family": "DejaVu Sans", "size": 10}
plt.rcParams.update({
    "font.family": FONT["family"],
    "font.size": FONT["size"],
    "axes.facecolor": PALETTE["background"],
    "figure.facecolor": "white",
    "axes.grid": True,
    "grid.color": PALETTE["grid"],
    "grid.linewidth": 0.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
})


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _save_or_show(fig: plt.Figure, save_path: Optional[str]) -> None:
    """Save figure to file or display it."""
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Chart saved: %s", save_path)
    else:
        plt.show()
    plt.close(fig)


def _pct_formatter(x, _):
    return f"{x:.0%}"


# ---------------------------------------------------------------------------
# 1. Efficient Frontier
# ---------------------------------------------------------------------------

def plot_efficient_frontier(
    frontier_df: pd.DataFrame,
    monte_carlo_df: Optional[pd.DataFrame] = None,
    special_portfolios: Optional[Dict[str, Tuple[float, float]]] = None,
    risk_free_rate: float = 0.02,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot the mean-variance efficient frontier.

    Parameters
    ----------
    frontier_df : pd.DataFrame
        Columns: Volatility, Return, Sharpe (from PortfolioOptimizer.efficient_frontier).
    monte_carlo_df : pd.DataFrame, optional
        Random portfolio cloud (from PortfolioOptimizer.monte_carlo_simulation).
    special_portfolios : dict, optional
        Named portfolios to highlight, e.g. {'Max Sharpe': (vol, ret)}.
    risk_free_rate : float
        For drawing the Capital Market Line.
    save_path : str, optional
        File path to save the chart.

    Returns
    -------
    matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Random portfolio cloud
    if monte_carlo_df is not None:
        scatter = ax.scatter(
            monte_carlo_df["Volatility"],
            monte_carlo_df["Return"],
            c=monte_carlo_df["Sharpe"],
            cmap="RdYlGn",
            alpha=0.3,
            s=8,
            zorder=1,
        )
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("Sharpe Ratio", fontsize=9)

    # Efficient frontier line
    frontier_clean = frontier_df.dropna(subset=["Volatility", "Return"]).sort_values("Volatility")
    ax.plot(
        frontier_clean["Volatility"],
        frontier_clean["Return"],
        color=PALETTE["primary"],
        linewidth=2.5,
        label="Efficient Frontier",
        zorder=3,
    )

    # Capital Market Line
    if not frontier_clean.empty:
        max_vol = frontier_clean["Volatility"].max() * 1.1
        best_sharpe_row = frontier_clean.loc[frontier_clean["Sharpe"].idxmax()]
        slope = (best_sharpe_row["Return"] - risk_free_rate) / best_sharpe_row["Volatility"]
        cml_x = np.array([0, max_vol])
        cml_y = risk_free_rate + slope * cml_x
        ax.plot(cml_x, cml_y, "--", color=PALETTE["neutral"], linewidth=1.2,
                label="Capital Market Line", zorder=2)

    # Special portfolios
    if special_portfolios:
        markers = ["*", "D", "^", "v", "s"]
        colors_sp = [PALETTE["highlight"], PALETTE["danger"], PALETTE["success"],
                     PALETTE["accent"], PALETTE["primary"]]
        for i, (name, (vol, ret)) in enumerate(special_portfolios.items()):
            ax.scatter(
                vol, ret,
                marker=markers[i % len(markers)],
                color=colors_sp[i % len(colors_sp)],
                s=200, zorder=5, label=name, edgecolors="black", linewidths=0.8,
            )

    ax.xaxis.set_major_formatter(mticker.FuncFormatter(_pct_formatter))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_pct_formatter))
    ax.set_xlabel("Annual Volatility", fontsize=11)
    ax.set_ylabel("Annual Expected Return", fontsize=11)
    ax.set_title("Efficient Frontier", fontsize=13)
    ax.legend(loc="upper left", fontsize=9)

    _save_or_show(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# 2. Portfolio Allocation
# ---------------------------------------------------------------------------

def plot_allocation(
    weights: pd.Series,
    title: str = "Portfolio Allocation",
    threshold: float = 0.005,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Dual panel: pie chart (left) and horizontal bar chart (right).

    Parameters
    ----------
    weights : pd.Series
        Asset weights indexed by ticker.
    title : str
        Chart title.
    threshold : float
        Minimum weight to display individually (smaller weights grouped as 'Other').
    save_path : str, optional
        Save path.

    Returns
    -------
    matplotlib Figure
    """
    weights = weights[weights.abs() > threshold]
    if weights.sum() < 0.9999:
        weights["Other"] = 1 - weights.sum()

    weights = weights.sort_values(ascending=False)
    colors = plt.cm.tab20(np.linspace(0, 1, len(weights)))

    fig, (ax_pie, ax_bar) = plt.subplots(1, 2, figsize=(12, 6))

    # Pie chart
    wedges, texts, autotexts = ax_pie.pie(
        weights.values,
        labels=None,
        autopct=lambda p: f"{p:.1f}%" if p > 2 else "",
        colors=colors,
        startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 1.2},
        pctdistance=0.8,
    )
    ax_pie.legend(
        wedges,
        weights.index,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=3,
        fontsize=8,
    )
    ax_pie.set_title("Allocation Pie", fontsize=12, fontweight="bold")

    # Horizontal bar chart
    bars = ax_bar.barh(
        weights.index,
        weights.values,
        color=colors,
        edgecolor="white",
        height=0.6,
    )
    for bar, v in zip(bars, weights.values):
        ax_bar.text(
            bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
            f"{v:.1%}", va="center", fontsize=9,
        )
    ax_bar.xaxis.set_major_formatter(mticker.FuncFormatter(_pct_formatter))
    ax_bar.set_xlabel("Weight", fontsize=11)
    ax_bar.set_title("Allocation Bar", fontsize=12, fontweight="bold")
    ax_bar.invert_yaxis()

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.01)
    _save_or_show(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# 3. Risk Contribution
# ---------------------------------------------------------------------------

def plot_risk_contribution(
    risk_contrib: pd.Series,
    weights: Optional[pd.Series] = None,
    title: str = "Risk Contribution",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Bar chart of per-asset risk contribution vs portfolio weight.

    Parameters
    ----------
    risk_contrib : pd.Series
        Fractional risk contribution per asset (sums to 1).
    weights : pd.Series, optional
        Portfolio weights to overlay for comparison.
    title : str
        Chart title.
    save_path : str, optional
        Save path.

    Returns
    -------
    matplotlib Figure
    """
    rc = risk_contrib.sort_values(ascending=False)
    x = np.arange(len(rc))
    width = 0.4

    fig, ax = plt.subplots(figsize=(10, 5))

    bars_rc = ax.bar(x, rc.values, width, label="Risk Contribution",
                     color=PALETTE["danger"], alpha=0.85, edgecolor="white")

    if weights is not None:
        w = weights.reindex(rc.index).fillna(0)
        ax.bar(x + width, w.values, width, label="Portfolio Weight",
               color=PALETTE["accent"], alpha=0.85, edgecolor="white")

    ax.set_xticks(x + width / 2 if weights is not None else x)
    ax.set_xticklabels(rc.index, rotation=30, ha="right", fontsize=9)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_pct_formatter))
    ax.set_ylabel("Fraction", fontsize=11)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=9)

    _save_or_show(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# 4. Drawdown Chart
# ---------------------------------------------------------------------------

def plot_drawdown(
    drawdown: pd.Series,
    title: str = "Portfolio Drawdown",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Shade the drawdown area below zero.

    Parameters
    ----------
    drawdown : pd.Series
        Drawdown series (negative values) from RiskMetrics.drawdown_series().
    title : str
        Chart title.
    save_path : str, optional
        Save path.

    Returns
    -------
    matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.fill_between(drawdown.index, drawdown.values, 0,
                    color=PALETTE["danger"], alpha=0.5, label="Drawdown")
    ax.plot(drawdown.index, drawdown.values, color=PALETTE["danger"],
            linewidth=0.8)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_pct_formatter))
    ax.set_ylabel("Drawdown", fontsize=11)
    ax.set_title(title, fontsize=13)

    # Annotate maximum drawdown
    idx_min = drawdown.idxmin()
    mdd = drawdown.min()
    ax.annotate(
        f"Max DD: {mdd:.1%}",
        xy=(idx_min, mdd),
        xytext=(idx_min, mdd * 0.5),
        arrowprops={"arrowstyle": "->", "color": PALETTE["primary"]},
        fontsize=9,
        color=PALETTE["primary"],
    )

    ax.axhline(0, color="black", linewidth=0.8)
    ax.legend(fontsize=9)
    _save_or_show(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# 5. Cumulative Performance
# ---------------------------------------------------------------------------

def plot_cumulative_performance(
    portfolio_returns: pd.Series,
    benchmark_returns: Optional[pd.Series] = None,
    title: str = "Cumulative Performance",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot cumulative wealth growth starting from 1.0.

    Parameters
    ----------
    portfolio_returns : pd.Series
        Daily portfolio returns.
    benchmark_returns : pd.Series, optional
        Benchmark daily returns for comparison.
    title : str
        Chart title.
    save_path : str, optional
        Save path.

    Returns
    -------
    matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(12, 5))

    cum_port = (1 + portfolio_returns).cumprod()
    ax.plot(cum_port.index, cum_port.values, color=PALETTE["primary"],
            linewidth=2, label=portfolio_returns.name or "Portfolio")

    if benchmark_returns is not None:
        cum_bench = (1 + benchmark_returns).cumprod()
        ax.plot(cum_bench.index, cum_bench.values, color=PALETTE["neutral"],
                linewidth=1.5, linestyle="--",
                label=benchmark_returns.name or "Benchmark")

    ax.axhline(1.0, color="black", linewidth=0.6, linestyle=":")
    ax.set_ylabel("Portfolio Value (base = 1)", fontsize=11)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=9)

    # Annotate final value
    final_val = float(cum_port.iloc[-1])
    ax.annotate(
        f"{final_val:.2f}x",
        xy=(cum_port.index[-1], final_val),
        xytext=(-40, 10),
        textcoords="offset points",
        fontsize=9,
        color=PALETTE["primary"],
    )

    _save_or_show(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# 6. Rolling Sharpe
# ---------------------------------------------------------------------------

def plot_rolling_sharpe(
    rolling_sharpe: pd.Series,
    title: str = "Rolling Sharpe Ratio (63-day)",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot rolling Sharpe ratio with shading for positive/negative periods.

    Parameters
    ----------
    rolling_sharpe : pd.Series
        Rolling Sharpe series from ReturnsCalculator.rolling_sharpe().
    title : str
        Chart title.
    save_path : str, optional
        Save path.

    Returns
    -------
    matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(12, 4))

    sharpe = rolling_sharpe.dropna()
    ax.plot(sharpe.index, sharpe.values, color=PALETTE["accent"],
            linewidth=1.5, label="Rolling Sharpe")

    ax.fill_between(sharpe.index, sharpe.values, 0,
                    where=sharpe.values >= 0,
                    alpha=0.3, color=PALETTE["success"], label="Positive")
    ax.fill_between(sharpe.index, sharpe.values, 0,
                    where=sharpe.values < 0,
                    alpha=0.3, color=PALETTE["danger"], label="Negative")

    ax.axhline(0, color="black", linewidth=0.8)
    ax.axhline(1, color=PALETTE["neutral"], linewidth=0.6, linestyle="--")
    ax.set_ylabel("Sharpe Ratio", fontsize=11)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=9)
    _save_or_show(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# 7. Correlation Heatmap
# ---------------------------------------------------------------------------

def plot_correlation_heatmap(
    corr: pd.DataFrame,
    title: str = "Asset Correlation Matrix",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Render a lower-triangle correlation heatmap.

    Parameters
    ----------
    corr : pd.DataFrame
        Correlation matrix.
    title : str
        Chart title.
    save_path : str, optional
        Save path.

    Returns
    -------
    matplotlib Figure
    """
    n = len(corr)
    fig, ax = plt.subplots(figsize=(max(6, n), max(5, n - 1)))

    cmap = LinearSegmentedColormap.from_list(
        "rg", [PALETTE["danger"], "white", PALETTE["success"]]
    )
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    data = corr.values.copy()
    data[mask] = np.nan

    im = ax.imshow(data, cmap=cmap, vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.03)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(corr.index, fontsize=9)

    for i in range(n):
        for j in range(i + 1):
            val = corr.iloc[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8,
                    color="black" if abs(val) < 0.7 else "white")

    ax.set_title(title, fontsize=13)
    _save_or_show(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# 8. Return Distribution with VaR / CVaR
# ---------------------------------------------------------------------------

def plot_return_distribution(
    port_returns: pd.Series,
    var_95: Optional[float] = None,
    cvar_95: Optional[float] = None,
    title: str = "Return Distribution",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Histogram of daily returns with VaR and CVaR overlaid.

    Parameters
    ----------
    port_returns : pd.Series
        Daily portfolio returns.
    var_95 : float, optional
        95 % VaR (positive number) to draw as a vertical line.
    cvar_95 : float, optional
        95 % CVaR (positive number) to draw as a vertical line.
    title : str
        Chart title.
    save_path : str, optional
        Save path.

    Returns
    -------
    matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.hist(port_returns.values, bins=60, color=PALETTE["accent"],
            alpha=0.7, edgecolor="white", density=True, label="Daily Returns")

    # Normal distribution overlay
    from scipy import stats as st
    mu, sigma = port_returns.mean(), port_returns.std()
    x = np.linspace(port_returns.min(), port_returns.max(), 300)
    ax.plot(x, st.norm.pdf(x, mu, sigma), color=PALETTE["primary"],
            linewidth=2, label="Normal fit")

    if var_95 is not None:
        ax.axvline(-var_95, color=PALETTE["highlight"], linewidth=2,
                   linestyle="--", label=f"VaR 95%: {var_95:.2%}")
    if cvar_95 is not None:
        ax.axvline(-cvar_95, color=PALETTE["danger"], linewidth=2,
                   linestyle="--", label=f"CVaR 95%: {cvar_95:.2%}")

    ax.xaxis.set_major_formatter(mticker.FuncFormatter(_pct_formatter))
    ax.set_xlabel("Daily Return", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=9)
    _save_or_show(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# 9. Full dashboard
# ---------------------------------------------------------------------------

def plot_dashboard(
    portfolio_returns: pd.Series,
    weights: pd.Series,
    risk_contrib: pd.Series,
    drawdown: pd.Series,
    benchmark_returns: Optional[pd.Series] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Render a 2×2 summary dashboard.

    Panels:
        Top-left    : Cumulative performance
        Top-right   : Portfolio allocation (bar)
        Bottom-left : Drawdown
        Bottom-right: Risk contribution

    Parameters
    ----------
    portfolio_returns : pd.Series
        Daily portfolio return series.
    weights : pd.Series
        Portfolio weights.
    risk_contrib : pd.Series
        Risk contribution per asset.
    drawdown : pd.Series
        Drawdown series.
    benchmark_returns : pd.Series, optional
        Benchmark returns for performance comparison.
    save_path : str, optional
        Save path.

    Returns
    -------
    matplotlib Figure
    """
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.3)

    # ---- Top-left: Cumulative performance ----
    ax1 = fig.add_subplot(gs[0, 0])
    cum_port = (1 + portfolio_returns).cumprod()
    ax1.plot(cum_port.index, cum_port.values, color=PALETTE["primary"], linewidth=2,
             label=portfolio_returns.name or "Portfolio")
    if benchmark_returns is not None:
        cum_bench = (1 + benchmark_returns).cumprod()
        ax1.plot(cum_bench.index, cum_bench.values, color=PALETTE["neutral"],
                 linewidth=1.5, linestyle="--",
                 label=benchmark_returns.name or "Benchmark")
    ax1.axhline(1.0, color="black", linewidth=0.5, linestyle=":")
    ax1.set_title("Cumulative Performance", fontsize=11)
    ax1.legend(fontsize=8)

    # ---- Top-right: Allocation bar ----
    ax2 = fig.add_subplot(gs[0, 1])
    w_sorted = weights.sort_values(ascending=False)
    colors = plt.cm.tab20(np.linspace(0, 1, len(w_sorted)))
    ax2.barh(w_sorted.index, w_sorted.values, color=colors,
             edgecolor="white", height=0.6)
    ax2.xaxis.set_major_formatter(mticker.FuncFormatter(_pct_formatter))
    ax2.set_title("Portfolio Allocation", fontsize=11)
    ax2.invert_yaxis()

    # ---- Bottom-left: Drawdown ----
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.fill_between(drawdown.index, drawdown.values, 0,
                     color=PALETTE["danger"], alpha=0.5)
    ax3.plot(drawdown.index, drawdown.values, color=PALETTE["danger"], linewidth=0.8)
    ax3.axhline(0, color="black", linewidth=0.5)
    ax3.yaxis.set_major_formatter(mticker.FuncFormatter(_pct_formatter))
    ax3.set_title("Drawdown", fontsize=11)

    # ---- Bottom-right: Risk contribution ----
    ax4 = fig.add_subplot(gs[1, 1])
    rc_sorted = risk_contrib.sort_values(ascending=False)
    x = np.arange(len(rc_sorted))
    width = 0.4
    ax4.bar(x, rc_sorted.values, width, label="Risk Contrib",
            color=PALETTE["danger"], alpha=0.85, edgecolor="white")
    w_aligned = weights.reindex(rc_sorted.index).fillna(0)
    ax4.bar(x + width, w_aligned.values, width, label="Weight",
            color=PALETTE["accent"], alpha=0.85, edgecolor="white")
    ax4.set_xticks(x + width / 2)
    ax4.set_xticklabels(rc_sorted.index, rotation=45, ha="right", fontsize=8)
    ax4.yaxis.set_major_formatter(mticker.FuncFormatter(_pct_formatter))
    ax4.set_title("Risk Contribution vs Weight", fontsize=11)
    ax4.legend(fontsize=8)

    fig.suptitle("Portfolio Analytics Dashboard", fontsize=15, fontweight="bold")
    _save_or_show(fig, save_path)
    return fig
