"""
main.py
=======
End-to-end demonstration of the Risk-Aware Portfolio Optimizer.

Pipeline
--------
1. Download 5 years of live market data (yfinance) for a diversified
   multi-asset universe
2. Compute log returns, covariance, and correlation matrices
3. Run all five optimization strategies with institutional constraints
4. Compute comprehensive risk metrics for each portfolio
5. Run a walk-forward backtest (monthly rebalancing, max-Sharpe strategy)
6. Generate a full set of professional charts

Run
---
    python main.py

Optional flags (edit CONFIG below):
    SAVE_CHARTS  – write PNGs to ./charts/
    RUN_BACKTEST – execute the walk-forward backtest (adds ~30 s)
"""

import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CONFIG = {
    "tickers": [
        "SPY",   # US equities (S&P 500 ETF)
        "EFA",   # International developed equities
        "EEM",   # Emerging market equities
        "AGG",   # US aggregate bonds
        "GLD",   # Gold
        "USO",   # Oil
        "BTC-USD",  # Bitcoin
        "VNQ",   # REITs
    ],
    "start_date": "2019-01-01",
    "end_date": "2024-01-01",
    "risk_free_rate": 0.04,       # current approximate US risk-free rate
    "max_weight": 0.35,           # max 35 % in any single asset
    "min_weight": 0.02,           # at least 2 % if asset is included
    "target_volatility": 0.15,    # 15 % annualised volatility cap
    "confidence_level": 0.95,
    "n_frontier_points": 50,
    "n_mc_portfolios": 3000,
    "rebalance_freq": "M",        # monthly backtest rebalancing
    "estimation_window": 252,     # 1-year rolling estimation window
    "save_charts": True,
    "charts_dir": "./charts",
    "run_backtest": True,
    "log_level": "INFO",
}

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=CONFIG["log_level"],
    format="%(asctime)s  %(levelname)-8s  %(name)s – %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------------

from data_loader import DataLoader
from returns_calculator import ReturnsCalculator
from risk_metrics import RiskMetrics, VaRMethod
from optimizer import PortfolioOptimizer, OptimizationStrategy
from constraints import ConstraintBuilder, describe_constraints, validate_weights
from backtest import BacktestEngine, RebalanceFrequency, make_optimizer_strategy
import visualization as viz


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def section(title: str) -> None:
    width = 60
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def print_weights(weights: pd.Series, label: str = "") -> None:
    if label:
        print(f"\n  {label}")
    for ticker, w in weights.items():
        bar = "█" * int(abs(w) * 40)
        sign = " " if w >= 0 else "-"
        print(f"    {ticker:<10} {sign}{abs(w):6.2%}  {bar}")


def print_risk_report(report: dict) -> None:
    for k, v in report.items():
        if isinstance(v, float):
            if any(x in k for x in ["Return", "Volatility", "Drawdown", "VaR", "CVaR"]):
                print(f"    {k:<40} {v:>8.2%}")
            else:
                print(f"    {k:<40} {v:>8.4f}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    charts_dir = Path(CONFIG["charts_dir"])
    if CONFIG["save_charts"]:
        charts_dir.mkdir(parents=True, exist_ok=True)

    def chart_path(name: str) -> str:
        return str(charts_dir / name) if CONFIG["save_charts"] else None

    # -----------------------------------------------------------------------
    # STEP 1 – Load data
    # -----------------------------------------------------------------------
    section("STEP 1 – Loading Market Data")

    loader = DataLoader(
        tickers=CONFIG["tickers"],
        start_date=CONFIG["start_date"],
        end_date=CONFIG["end_date"],
    )

    try:
        prices = loader.from_yfinance()
        prices = loader.trim_to_common_history()
        logger.info("Price data shape: %s", prices.shape)
        print(f"\n  Assets loaded    : {list(prices.columns)}")
        print(f"  Date range       : {prices.index[0].date()} → {prices.index[-1].date()}")
        print(f"  Trading days     : {len(prices)}")
    except Exception as exc:
        logger.error("Failed to download data: %s", exc)
        logger.info("Generating synthetic price data for demonstration …")
        prices = _generate_synthetic_prices(
            tickers=CONFIG["tickers"],
            start=CONFIG["start_date"],
            end=CONFIG["end_date"],
        )
        loader.prices = prices

    # -----------------------------------------------------------------------
    # STEP 2 – Compute return statistics
    # -----------------------------------------------------------------------
    section("STEP 2 – Return Statistics")

    calc = ReturnsCalculator(prices)
    mu = calc.annualized_returns()
    cov = calc.covariance_matrix()
    corr = calc.correlation_matrix()

    print("\n  Asset return / volatility summary:")
    print(calc.summary().to_string(
        float_format=lambda x: f"{x:.2%}"
    ))

    # -----------------------------------------------------------------------
    # STEP 3 – Build constraints
    # -----------------------------------------------------------------------
    section("STEP 3 – Institutional Constraints")

    n_assets = len(CONFIG["tickers"])
    asset_names = list(prices.columns)

    cb = (ConstraintBuilder(n_assets, asset_names)
          .long_only()
          .max_weight(CONFIG["max_weight"])
          .min_weight(CONFIG["min_weight"])
          .target_volatility(CONFIG["target_volatility"]))

    # Example sector constraint: crypto ≤ 10 %
    crypto_assets = [t for t in asset_names if "BTC" in t or "ETH" in t]
    if crypto_assets:
        cb.add_sector_constraint(crypto_assets, max_exposure=0.10)

    constraints = cb.build()
    print(describe_constraints(constraints, asset_names))

    # -----------------------------------------------------------------------
    # STEP 4 – Run all optimization strategies
    # -----------------------------------------------------------------------
    section("STEP 4 – Portfolio Optimization")

    optimizer = PortfolioOptimizer(
        mu=mu,
        cov=cov,
        returns=calc.log_returns,
        risk_free_rate=CONFIG["risk_free_rate"],
    )

    strategies = [
        OptimizationStrategy.MAX_SHARPE,
        OptimizationStrategy.MIN_VOLATILITY,
        OptimizationStrategy.MEAN_VARIANCE,
        OptimizationStrategy.MEAN_CVAR,
        OptimizationStrategy.RISK_PARITY,
        OptimizationStrategy.HRP,
    ]

    results = {}
    for strat in strategies:
        try:
            result = optimizer.optimize(strategy=strat, constraints=constraints)
            results[strat.value] = result
            print(f"\n  ── {result.strategy.upper()} ──")
            print_weights(result.weights_series())
            print(f"\n  {result.summary().to_string()}")
        except Exception as exc:
            logger.warning("Strategy %s failed: %s", strat.value, exc)

    # -----------------------------------------------------------------------
    # STEP 5 – Risk metrics for best portfolio (Max Sharpe)
    # -----------------------------------------------------------------------
    section("STEP 5 – Risk Metrics (Max Sharpe Portfolio)")

    best = results.get(OptimizationStrategy.MAX_SHARPE.value)
    if best is None:
        best = list(results.values())[0]

    rm = RiskMetrics(
        returns=calc.simple_returns,
        weights=best.weights,
    )
    report = rm.full_report(CONFIG["risk_free_rate"])
    print_risk_report(report)

    # Stress test
    print("\n  Stress Test Results:")
    stress = rm.stress_test()
    print(stress.to_string(float_format=lambda x: f"{x:.2f}"))

    # Validate constraints
    valid, violations = validate_weights(best.weights, constraints)
    print(f"\n  Constraints satisfied: {valid}")
    if violations:
        for v in violations:
            print(f"    VIOLATION: {v}")

    # -----------------------------------------------------------------------
    # STEP 6 – Efficient Frontier + Monte Carlo
    # -----------------------------------------------------------------------
    section("STEP 6 – Efficient Frontier")

    frontier_df = optimizer.efficient_frontier(
        n_points=CONFIG["n_frontier_points"],
        constraints=constraints,
    )
    mc_df = optimizer.monte_carlo_simulation(
        n_portfolios=CONFIG["n_mc_portfolios"]
    )
    print(f"  Frontier computed: {len(frontier_df)} points")
    print(f"  Monte Carlo: {len(mc_df)} portfolios")

    # Special portfolios to highlight
    special = {}
    for strat_name, res in results.items():
        special[strat_name.replace("_", " ").title()] = (
            res.volatility, res.expected_return
        )

    viz.plot_efficient_frontier(
        frontier_df,
        monte_carlo_df=mc_df,
        special_portfolios=special,
        risk_free_rate=CONFIG["risk_free_rate"],
        save_path=chart_path("01_efficient_frontier.png"),
    )

    # -----------------------------------------------------------------------
    # STEP 7 – Visualizations for Max Sharpe portfolio
    # -----------------------------------------------------------------------
    section("STEP 7 – Visualizations")

    best_weights_series = best.weights_series()
    rc = rm.risk_contribution()
    port_returns = calc.simple_returns @ best.weights
    port_returns.name = "Max Sharpe Portfolio"
    drawdown = rm.drawdown_series()

    # Allocation
    viz.plot_allocation(
        best_weights_series,
        title="Max Sharpe Portfolio Allocation",
        save_path=chart_path("02_allocation.png"),
    )

    # Risk contribution
    viz.plot_risk_contribution(
        rc, weights=best_weights_series,
        title="Risk Contribution vs Weight",
        save_path=chart_path("03_risk_contribution.png"),
    )

    # Drawdown
    viz.plot_drawdown(
        drawdown,
        save_path=chart_path("04_drawdown.png"),
    )

    # Cumulative performance (use SPY as benchmark if available)
    bench_ret = None
    if "SPY" in prices.columns:
        bench_ret = calc.simple_returns["SPY"].rename("SPY (Benchmark)")

    viz.plot_cumulative_performance(
        port_returns,
        benchmark_returns=bench_ret,
        save_path=chart_path("05_cumulative_performance.png"),
    )

    # Rolling Sharpe
    rolling_sharpe = calc.rolling_sharpe(
        best.weights,
        risk_free_rate=CONFIG["risk_free_rate"],
    )
    viz.plot_rolling_sharpe(
        rolling_sharpe,
        save_path=chart_path("06_rolling_sharpe.png"),
    )

    # Correlation heatmap
    viz.plot_correlation_heatmap(
        corr,
        save_path=chart_path("07_correlation_heatmap.png"),
    )

    # Return distribution with VaR / CVaR
    viz.plot_return_distribution(
        port_returns,
        var_95=rm.var(0.95, VaRMethod.HISTORICAL),
        cvar_95=rm.cvar(0.95, VaRMethod.HISTORICAL),
        save_path=chart_path("08_return_distribution.png"),
    )

    # Dashboard
    viz.plot_dashboard(
        portfolio_returns=port_returns,
        weights=best_weights_series,
        risk_contrib=rc,
        drawdown=drawdown,
        benchmark_returns=bench_ret,
        save_path=chart_path("09_dashboard.png"),
    )

    print(f"  Charts saved to: {charts_dir.resolve()}")

    # -----------------------------------------------------------------------
    # STEP 8 – Backtest
    # -----------------------------------------------------------------------
    if CONFIG["run_backtest"]:
        section("STEP 8 – Walk-Forward Backtest (Max Sharpe, Monthly Rebalancing)")

        def optimizer_factory(returns_window: pd.DataFrame) -> PortfolioOptimizer:
            """Build a fresh optimizer from a returns window."""
            from returns_calculator import ReturnsCalculator as RC
            rc_inner = RC(
                pd.DataFrame(
                    np.exp(returns_window.cumsum().values),  # back to prices
                    columns=returns_window.columns,
                )
            )
            # Use the window returns directly for covariance
            mu_w = returns_window.mean() * 252
            cov_w = returns_window.cov() * 252
            return PortfolioOptimizer(
                mu=pd.Series(mu_w.values, index=returns_window.columns),
                cov=pd.DataFrame(cov_w.values, index=returns_window.columns, columns=returns_window.columns),
                returns=returns_window,
                risk_free_rate=CONFIG["risk_free_rate"],
            )

        bt_constraints = (ConstraintBuilder(n_assets, asset_names)
                          .long_only()
                          .max_weight(CONFIG["max_weight"])
                          .build())

        strategy_fn = make_optimizer_strategy(
            optimizer_factory,
            strategy_name="max_sharpe",
            constraints=bt_constraints,
        )

        bench_prices = prices["SPY"] if "SPY" in prices.columns else None

        engine = BacktestEngine(
            prices=prices,
            strategy_fn=strategy_fn,
            rebalance_freq=RebalanceFrequency(CONFIG["rebalance_freq"]),
            estimation_window=CONFIG["estimation_window"],
            benchmark_prices=bench_prices,
        )

        bt_result = engine.run()
        bt_result.print_summary()

        # Backtest performance chart
        viz.plot_cumulative_performance(
            bt_result.portfolio_returns,
            benchmark_returns=bt_result.benchmark_returns,
            title="Backtest: Cumulative Performance (Max Sharpe, Monthly Rebalancing)",
            save_path=chart_path("10_backtest_performance.png"),
        )

        # Backtest drawdown
        bt_rm = RiskMetrics(
            returns=pd.DataFrame(bt_result.portfolio_returns),
            weights=np.array([1.0]),
        )
        viz.plot_drawdown(
            bt_rm.drawdown_series(),
            title="Backtest: Portfolio Drawdown",
            save_path=chart_path("11_backtest_drawdown.png"),
        )

    # -----------------------------------------------------------------------
    # Done
    # -----------------------------------------------------------------------
    section("COMPLETE")
    print("\n  Risk-Aware Portfolio Optimizer ran successfully.\n")

    if CONFIG["save_charts"]:
        chart_files = sorted(charts_dir.glob("*.png"))
        print(f"  {len(chart_files)} charts saved:")
        for f in chart_files:
            print(f"    {f.name}")
    print()


# ---------------------------------------------------------------------------
# Synthetic data fallback
# ---------------------------------------------------------------------------

def _generate_synthetic_prices(
    tickers: list,
    start: str = "2019-01-01",
    end: str = "2024-01-01",
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate correlated synthetic price series when live data is unavailable.
    Used as a fallback for offline / CI environments.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, end=end)
    n, t = len(tickers), len(dates)

    # Annual return & vol assumptions per asset class
    ann_rets = np.array([0.10, 0.08, 0.09, 0.03, 0.05, 0.04, 0.30, 0.07])[:n]
    ann_vols = np.array([0.16, 0.18, 0.22, 0.05, 0.15, 0.30, 0.70, 0.18])[:n]

    daily_rets = ann_rets / 252
    daily_vols = ann_vols / np.sqrt(252)

    # Correlation structure
    corr_mat = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            corr_mat[i, j] = corr_mat[j, i] = 0.4 if i < 4 and j < 4 else 0.1

    # Cholesky-correlated returns
    L = np.linalg.cholesky(corr_mat + 1e-6 * np.eye(n))
    z = rng.standard_normal((t, n))
    returns = z @ L.T * daily_vols + daily_rets

    # Convert to price series
    prices = pd.DataFrame(
        100 * np.exp(np.cumsum(returns, axis=0)),
        index=dates,
        columns=tickers[:n],
    )
    logger.info("Generated synthetic price data: %s", prices.shape)
    return prices


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        sys.exit(0)
    except Exception as exc:
        logger.exception("Fatal error: %s", exc)
        sys.exit(1)
