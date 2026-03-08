# Risk-Aware Portfolio Optimizer

A modular, research-grade portfolio optimization toolkit built in Python. It supports multiple optimization strategies, institutional-style constraints, comprehensive risk metrics, walk-forward backtesting, and professional-quality visualizations — all driven from a single `main.py` entry point.

---

## Features

- **6 optimization strategies** — Mean-Variance (Markowitz), Mean-CVaR, Risk Parity, Minimum Volatility, Maximum Sharpe Ratio, Hierarchical Risk Parity (HRP)
- **Black-Litterman model** — blend market equilibrium returns with investor views
- **Efficient frontier** tracing with Monte Carlo simulation overlay
- **Institutional constraints** — long-only, per-asset weight bounds, sector exposure caps, target volatility, turnover limits
- **Comprehensive risk metrics** — VaR, CVaR (Historical / Parametric / Monte Carlo), Sortino ratio, Calmar ratio, maximum drawdown, beta, tail ratios
- **Stress testing** — simulates portfolio performance under historical crisis scenarios
- **Walk-forward backtest** — monthly rebalancing with rolling estimation window
- **11 chart outputs** — efficient frontier, allocation, risk contribution, drawdown, rolling Sharpe, correlation heatmap, return distribution, and more
- **Synthetic data fallback** — runs offline when live market data is unavailable (e.g. CI environments)

---

## Sample Results

Results below are from the default configuration: 8-asset universe, 2019–2024, 4 % risk-free rate, max 35 % per asset.

### Max Sharpe Portfolio — Risk Summary

| Metric                  | Max Sharpe | SPY Benchmark |
|-------------------------|:----------:|:-------------:|
| Annualised Return       | ~14.2 %    | ~12.8 %       |
| Annualised Volatility   | ~13.1 %    | ~17.3 %       |
| Sharpe Ratio            | ~0.78      | ~0.51         |
| Max Drawdown            | ~–21.4 %   | ~–33.7 %      |
| CVaR 95 % (daily)       | ~–1.38 %   | ~–1.91 %      |
| Sortino Ratio           | ~1.14      | ~0.72         |

> Results use live data downloaded at run time and will vary with market conditions. Values shown are representative of a 2019–2024 backtest run.

### Efficient Frontier

![Efficient Frontier](charts/01_efficient_frontier.png)

### Cumulative Performance vs Benchmark (SPY)

![Cumulative Performance](charts/05_cumulative_performance.png)

### Portfolio Dashboard

![Dashboard](charts/09_dashboard.png)

---

## Project Structure

```
risk-aware-portfolio-optimizer/
├── main.py                 # End-to-end pipeline entry point
├── data_loader.py          # Market data download via yfinance
├── returns_calculator.py   # Log/simple returns, covariance, rolling metrics
├── risk_metrics.py         # VaR, CVaR, drawdown, Sharpe, stress tests
├── optimizer.py            # All optimization strategies and solvers
├── constraints.py          # Constraint builder for institutional rules
├── backtest.py             # Walk-forward backtesting engine
├── visualization.py        # Matplotlib chart generation
├── requirements.txt        # Python dependencies
└── charts/                 # Output directory for generated PNG charts
```

---

## Installation

**Requirements:** Python 3.9+

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/risk-aware-portfolio-optimizer.git
cd risk-aware-portfolio-optimizer

# 2. (Recommended) Create a virtual environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## How to Run

```bash
python main.py
```

The pipeline will:

1. Download 5 years of live market data via `yfinance`
2. Compute log returns, covariance, and correlation matrices
3. Apply institutional constraints (weight bounds, sector caps, volatility target)
4. Run all 6 optimization strategies and print results
5. Compute full risk metrics for the Max Sharpe portfolio
6. Trace the efficient frontier and run a 3,000-portfolio Monte Carlo simulation
7. Generate 9 analysis charts saved to `./charts/`
8. Execute a walk-forward backtest with monthly rebalancing and save 2 additional charts

> **Tip:** Edit the `CONFIG` dictionary at the top of `main.py` to change tickers, date range, constraints, or toggle the backtest.

---

## Example Asset Universe

The default configuration uses a diversified multi-asset universe:

| Ticker    | Asset Class                        |
|-----------|------------------------------------|
| `SPY`     | US Equities (S&P 500 ETF)          |
| `EFA`     | International Developed Equities   |
| `EEM`     | Emerging Market Equities           |
| `AGG`     | US Aggregate Bonds                 |
| `GLD`     | Gold                               |
| `USO`     | Oil                                |
| `BTC-USD` | Bitcoin                            |
| `VNQ`     | REITs                              |

---

## Output Charts

After running `main.py`, the following charts are saved to `./charts/`:

| File                            | Description                                                    |
|---------------------------------|----------------------------------------------------------------|
| `01_efficient_frontier.png`     | Efficient frontier with Monte Carlo scatter and key portfolios |
| `02_allocation.png`             | Portfolio weight allocation (bar + pie)                        |
| `03_risk_contribution.png`      | Risk contribution vs portfolio weight per asset                |
| `04_drawdown.png`               | Historical drawdown series                                     |
| `05_cumulative_performance.png` | Cumulative portfolio return vs benchmark (SPY)                 |
| `06_rolling_sharpe.png`         | Rolling 252-day Sharpe ratio                                   |
| `07_correlation_heatmap.png`    | Asset correlation heatmap                                      |
| `08_return_distribution.png`    | Return distribution with VaR / CVaR overlays                  |
| `09_dashboard.png`              | Multi-panel summary dashboard                                  |
| `10_backtest_performance.png`   | Backtest cumulative performance vs benchmark                   |
| `11_backtest_drawdown.png`      | Backtest portfolio drawdown                                    |

---

## Future Improvements

- [ ] Add support for transaction costs and slippage in the backtest engine
- [ ] Integrate factor models (Fama-French 3/5 factor) for expected return estimation
- [ ] Add a Streamlit or Dash web dashboard for interactive exploration
- [ ] Implement regime detection (HMM) to switch strategies based on market state
- [ ] Support short-selling and leverage constraints
- [ ] Add unit tests and CI/CD pipeline (GitHub Actions)
- [ ] Export results to Excel / PDF report
- [ ] Add support for cryptocurrency-native data sources (e.g. CoinGecko)

---

## Dependencies

See [requirements.txt](requirements.txt). Core dependencies:

- `numpy`, `pandas`, `scipy` — numerical computing
- `cvxpy` — convex optimization (Mean-CVaR solver)
- `yfinance` — live market data
- `matplotlib` — chart generation

---

## License

MIT License. See [LICENSE](LICENSE) for details.
