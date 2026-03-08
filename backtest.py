"""
backtest.py
===========
Walk-forward backtesting engine for portfolio strategies.

Features
--------
- Monthly or quarterly rebalancing
- Rolling estimation window (expanding or fixed)
- Benchmark comparison
- Full performance attribution
- Risk metrics computed through time
- Stress-period analysis

The BacktestEngine accepts any callable strategy function
(strategy_fn(returns_window) → np.ndarray of weights) so it works with
every optimizer strategy defined in optimizer.py.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class RebalanceFrequency(str, Enum):
    MONTHLY = "M"
    QUARTERLY = "Q"
    SEMI_ANNUAL = "6M"
    ANNUAL = "A"


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class BacktestResult:
    """Carries all outputs of a backtest run."""

    portfolio_returns: pd.Series
    benchmark_returns: Optional[pd.Series]
    weights_history: pd.DataFrame           # date → asset weights
    turnover_history: pd.Series
    metrics_history: pd.DataFrame           # rolling risk metrics
    summary: Dict = field(default_factory=dict)

    def __post_init__(self):
        self.summary = self._compute_summary()

    def _compute_summary(self) -> Dict:
        ret = self.portfolio_returns
        ann_ret = ret.mean() * TRADING_DAYS
        ann_vol = ret.std() * np.sqrt(TRADING_DAYS)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan

        cum = (1 + ret).cumprod()
        rolling_max = cum.cummax()
        dd = (cum - rolling_max) / rolling_max
        mdd = float(-dd.min())

        summary = {
            "Total Return": float((1 + ret).prod() - 1),
            "Annualized Return": float(ann_ret),
            "Annualized Volatility": float(ann_vol),
            "Sharpe Ratio": float(sharpe),
            "Max Drawdown": mdd,
            "Calmar Ratio": float(ann_ret / mdd) if mdd > 0 else np.inf,
            "Average Turnover": float(self.turnover_history.mean()),
        }

        if self.benchmark_returns is not None:
            bret = self.benchmark_returns.reindex(ret.index).fillna(0)
            active = ret - bret
            ann_active = active.mean() * TRADING_DAYS
            te = active.std() * np.sqrt(TRADING_DAYS)
            ir = ann_active / te if te > 0 else np.nan

            bcum = (1 + bret).cumprod()
            b_mdd = float(-((bcum - bcum.cummax()) / bcum.cummax()).min())

            summary.update({
                "Benchmark Annualized Return": float(bret.mean() * TRADING_DAYS),
                "Benchmark Max Drawdown": b_mdd,
                "Active Return (Alpha)": float(ann_active),
                "Tracking Error": float(te),
                "Information Ratio": float(ir),
            })

        return summary

    def print_summary(self) -> None:
        """Pretty-print the backtest summary."""
        print("\n" + "=" * 55)
        print("  BACKTEST PERFORMANCE SUMMARY")
        print("=" * 55)
        for k, v in self.summary.items():
            if isinstance(v, float):
                fmt = f"{v:.2%}" if "Return" in k or "Drawdown" in k or "Alpha" in k or "Error" in k else f"{v:.4f}"
                print(f"  {k:<35} {fmt}")
            else:
                print(f"  {k:<35} {v}")
        print("=" * 55 + "\n")


# ---------------------------------------------------------------------------
# Backtest engine
# ---------------------------------------------------------------------------

class BacktestEngine:
    """
    Walk-forward backtesting engine.

    Parameters
    ----------
    prices : pd.DataFrame
        Daily asset prices (rows = dates, columns = assets).
    strategy_fn : callable
        Function signature: ``(returns_window: pd.DataFrame) → np.ndarray``
        Given a window of historical returns, returns portfolio weights.
    rebalance_freq : RebalanceFrequency
        How often to rebalance (default monthly).
    estimation_window : int or None
        Number of trading days in the lookback window used for estimation.
        If None, uses an expanding window from the start.
    benchmark_prices : pd.Series, optional
        Benchmark price series for comparison.
    trading_days : int
        Annualisation factor.
    """

    def __init__(
        self,
        prices: pd.DataFrame,
        strategy_fn: Callable[[pd.DataFrame], np.ndarray],
        rebalance_freq: RebalanceFrequency = RebalanceFrequency.MONTHLY,
        estimation_window: Optional[int] = 252,
        benchmark_prices: Optional[pd.Series] = None,
        trading_days: int = TRADING_DAYS,
    ):
        self.prices = prices.copy()
        self.strategy_fn = strategy_fn
        self.rebalance_freq = RebalanceFrequency(rebalance_freq)
        self.estimation_window = estimation_window
        self.benchmark_prices = benchmark_prices
        self.trading_days = trading_days
        self.n_assets = prices.shape[1]
        self.asset_names = list(prices.columns)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self, start_date: Optional[str] = None) -> BacktestResult:
        """
        Execute the walk-forward backtest.

        Parameters
        ----------
        start_date : str, optional
            Date from which to start trading. Defaults to the earliest date
            permitted by the estimation window.

        Returns
        -------
        BacktestResult
        """
        # Simple returns for portfolio simulation
        returns = self.prices.pct_change().dropna()

        # Determine warm-up period
        if self.estimation_window is not None:
            min_start_idx = self.estimation_window
        else:
            min_start_idx = 63  # minimum 1 quarter

        if start_date is not None:
            start_dt = pd.Timestamp(start_date)
            min_start_idx = max(
                min_start_idx,
                returns.index.searchsorted(start_dt),
            )

        if min_start_idx >= len(returns):
            raise ValueError("Not enough data for the specified estimation window.")

        # Build rebalance dates
        rebalance_dates = self._get_rebalance_dates(
            returns.index[min_start_idx:]
        )

        logger.info(
            "Backtest: %d rebalance dates from %s to %s",
            len(rebalance_dates),
            rebalance_dates[0].date(),
            rebalance_dates[-1].date(),
        )

        # Storage
        weights_record: Dict[pd.Timestamp, np.ndarray] = {}
        current_weights = np.ones(self.n_assets) / self.n_assets

        for rebal_date in rebalance_dates:
            # Estimation window
            idx = returns.index.searchsorted(rebal_date)
            if self.estimation_window is not None:
                window_start = max(0, idx - self.estimation_window)
            else:
                window_start = 0
            window = returns.iloc[window_start:idx]

            if window.empty or len(window) < 20:
                logger.debug("Skipping rebalance at %s: insufficient window.", rebal_date)
                weights_record[rebal_date] = current_weights
                continue

            try:
                new_weights = self.strategy_fn(window)
                new_weights = np.clip(new_weights, 0, None)
                total = new_weights.sum()
                current_weights = new_weights / total if total > 0 else current_weights
            except Exception as exc:
                logger.warning("Strategy failed at %s: %s. Holding.", rebal_date, exc)

            weights_record[rebal_date] = current_weights.copy()

        # Simulate returns using daily weights (held between rebalances)
        port_returns, actual_weights, turnovers = self._simulate(
            returns, weights_record, min_start_idx
        )

        # Benchmark
        bench_returns = None
        if self.benchmark_prices is not None:
            bench_returns = self.benchmark_prices.pct_change().dropna()
            bench_returns = bench_returns.reindex(port_returns.index).fillna(0)
            bench_returns.name = "Benchmark"

        # Rolling risk metrics
        metrics_history = self._rolling_metrics(port_returns)

        result = BacktestResult(
            portfolio_returns=port_returns,
            benchmark_returns=bench_returns,
            weights_history=actual_weights,
            turnover_history=turnovers,
            metrics_history=metrics_history,
        )

        logger.info("Backtest complete. Total return: %.2f%%", result.summary["Total Return"] * 100)
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_rebalance_dates(self, date_range: pd.DatetimeIndex) -> pd.DatetimeIndex:
        """Return end-of-period dates for rebalancing."""
        freq_map = {
            RebalanceFrequency.MONTHLY: "ME",
            RebalanceFrequency.QUARTERLY: "QE",
            RebalanceFrequency.SEMI_ANNUAL: "6ME",
            RebalanceFrequency.ANNUAL: "YE",
        }
        freq = freq_map[self.rebalance_freq]
        period_ends = pd.date_range(
            start=date_range[0], end=date_range[-1], freq=freq
        )
        # Snap to actual trading dates
        rebal = []
        for pe in period_ends:
            idx = date_range.searchsorted(pe, side="right")
            if idx < len(date_range):
                rebal.append(date_range[idx])
            else:
                rebal.append(date_range[-1])
        return pd.DatetimeIndex(sorted(set(rebal)))

    def _simulate(
        self,
        returns: pd.DataFrame,
        weights_record: Dict[pd.Timestamp, np.ndarray],
        start_idx: int,
    ):
        """
        Simulate daily portfolio returns given recorded rebalance weights.

        Returns
        -------
        (portfolio_returns, weights_df, turnover_series)
        """
        trading_dates = returns.index[start_idx:]
        sorted_rebal = sorted(weights_record.keys())

        port_rets = []
        all_weights = []
        turnovers = []
        current_w = np.ones(self.n_assets) / self.n_assets
        prev_w = current_w.copy()

        # Pointer into sorted rebalance dates
        rebal_ptr = 0

        for date in trading_dates:
            # Advance to next rebalance if needed
            while rebal_ptr < len(sorted_rebal) and sorted_rebal[rebal_ptr] <= date:
                new_w = weights_record[sorted_rebal[rebal_ptr]]
                turnover = float(np.sum(np.abs(new_w - prev_w)))
                turnovers.append((sorted_rebal[rebal_ptr], turnover))
                prev_w = current_w.copy()
                current_w = new_w
                rebal_ptr += 1

            day_ret = returns.loc[date].values
            port_ret = float(current_w @ day_ret)
            port_rets.append(port_ret)
            all_weights.append(current_w.copy())

        port_series = pd.Series(port_rets, index=trading_dates, name="Portfolio")
        weights_df = pd.DataFrame(
            all_weights, index=trading_dates, columns=self.asset_names
        )
        if turnovers:
            to_dates, to_vals = zip(*turnovers)
            turnover_series = pd.Series(to_vals, index=to_dates, name="Turnover")
        else:
            turnover_series = pd.Series(dtype=float, name="Turnover")

        return port_series, weights_df, turnover_series

    @staticmethod
    def _rolling_metrics(
        port_returns: pd.Series,
        window: int = 63,
    ) -> pd.DataFrame:
        """Compute rolling annualised return, vol, and Sharpe."""
        ann = TRADING_DAYS
        roll_ret = port_returns.rolling(window).mean() * ann
        roll_vol = port_returns.rolling(window).std() * np.sqrt(ann)
        roll_sharpe = roll_ret / roll_vol

        return pd.DataFrame(
            {
                "Rolling Return": roll_ret,
                "Rolling Volatility": roll_vol,
                "Rolling Sharpe": roll_sharpe,
            }
        )


# ---------------------------------------------------------------------------
# Strategy wrappers (convenience functions)
# ---------------------------------------------------------------------------

def equal_weight_strategy(returns_window: pd.DataFrame) -> np.ndarray:
    """Simple equal-weight strategy."""
    n = returns_window.shape[1]
    return np.ones(n) / n


def inverse_vol_strategy(returns_window: pd.DataFrame) -> np.ndarray:
    """Inverse-volatility weighting."""
    vol = returns_window.std()
    inv_vol = 1.0 / vol.replace(0, np.nan).fillna(vol.max())
    return (inv_vol / inv_vol.sum()).values


def make_optimizer_strategy(
    optimizer_factory: Callable,
    strategy_name: str = "max_sharpe",
    constraints: Optional[Dict] = None,
) -> Callable:
    """
    Wrap a PortfolioOptimizer into a strategy callable for BacktestEngine.

    Parameters
    ----------
    optimizer_factory : callable
        A function ``(returns_window) → PortfolioOptimizer``.
    strategy_name : str
        Name of the OptimizationStrategy to run.
    constraints : dict, optional
        Constraint dict to pass to optimize().

    Returns
    -------
    callable
        Strategy function suitable for BacktestEngine.
    """
    def strategy(returns_window: pd.DataFrame) -> np.ndarray:
        opt = optimizer_factory(returns_window)
        result = opt.optimize(strategy=strategy_name, constraints=constraints or {})
        return result.weights

    return strategy
