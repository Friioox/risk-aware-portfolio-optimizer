"""
risk_metrics.py
===============
Computes a comprehensive suite of portfolio risk metrics:

1. Variance / Volatility
2. Value at Risk (VaR)  — historical, parametric, Monte Carlo
3. Conditional Value at Risk (CVaR / Expected Shortfall)
4. Maximum Drawdown
5. Risk Contribution per asset
6. Calmar Ratio, Sortino Ratio
7. Stress-testing scenarios

All metrics accept portfolio weights plus a returns DataFrame (or pre-computed
portfolio return series) and return consistent scalar or pd.Series outputs.
"""

import logging
from enum import Enum
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

TRADING_DAYS = 252


class VaRMethod(str, Enum):
    HISTORICAL = "historical"
    PARAMETRIC = "parametric"
    MONTE_CARLO = "monte_carlo"


class RiskMetrics:
    """
    Computes risk metrics for a portfolio.

    Parameters
    ----------
    returns : pd.DataFrame
        Daily asset returns (rows = dates, columns = assets).
    weights : np.ndarray
        Portfolio weights (1-D, must align with columns of ``returns``).
    trading_days : int
        Annualisation factor (default 252).
    """

    def __init__(
        self,
        returns: pd.DataFrame,
        weights: np.ndarray,
        trading_days: int = TRADING_DAYS,
    ):
        if returns.empty:
            raise ValueError("Returns DataFrame is empty.")
        if len(weights) != returns.shape[1]:
            raise ValueError(
                f"Weight vector length ({len(weights)}) does not match "
                f"number of assets ({returns.shape[1]})."
            )
        self.returns = returns.copy()
        self.weights = np.asarray(weights, dtype=float)
        self.trading_days = trading_days

        # Pre-compute portfolio daily return series (simple)
        self._port_returns: pd.Series = (self.returns @ self.weights)
        self._port_returns.name = "Portfolio"

    # ------------------------------------------------------------------
    # Basic statistics
    # ------------------------------------------------------------------

    def annualized_return(self) -> float:
        """Annualised expected portfolio return."""
        return float(self._port_returns.mean() * self.trading_days)

    def annualized_volatility(self) -> float:
        """Annualised portfolio standard deviation."""
        return float(self._port_returns.std() * np.sqrt(self.trading_days))

    def annualized_variance(self) -> float:
        """Annualised portfolio variance."""
        return float(self._port_returns.var() * self.trading_days)

    def sharpe_ratio(self, risk_free_rate: float = 0.02) -> float:
        """Annualised Sharpe ratio."""
        vol = self.annualized_volatility()
        if vol == 0:
            return -np.inf
        return (self.annualized_return() - risk_free_rate) / vol

    def sortino_ratio(self, risk_free_rate: float = 0.02) -> float:
        """
        Sortino ratio: excess return divided by downside deviation.

        Only negative returns contribute to downside deviation.
        """
        daily_rf = risk_free_rate / self.trading_days
        excess = self._port_returns - daily_rf
        downside = excess[excess < 0]
        downside_std = downside.std() * np.sqrt(self.trading_days)
        if downside_std == 0:
            return np.inf
        return self.annualized_return() / downside_std

    # ------------------------------------------------------------------
    # Value at Risk
    # ------------------------------------------------------------------

    def var(
        self,
        confidence: float = 0.95,
        method: Union[VaRMethod, str] = VaRMethod.HISTORICAL,
        horizon: int = 1,
        n_simulations: int = 10_000,
    ) -> float:
        """
        Portfolio Value at Risk (positive number = loss).

        Parameters
        ----------
        confidence : float
            Confidence level (e.g. 0.95 for 95 % VaR).
        method : VaRMethod or str
            ``'historical'``, ``'parametric'``, or ``'monte_carlo'``.
        horizon : int
            Holding period in days (default 1). Scaled via square-root-of-time.
        n_simulations : int
            Number of Monte Carlo paths (only used when method = 'monte_carlo').

        Returns
        -------
        float
            VaR as a positive decimal (e.g. 0.02 = 2 % loss).
        """
        method = VaRMethod(method)
        alpha = 1 - confidence

        if method == VaRMethod.HISTORICAL:
            var_1d = -np.percentile(self._port_returns, alpha * 100)

        elif method == VaRMethod.PARAMETRIC:
            mu = self._port_returns.mean()
            sigma = self._port_returns.std()
            z = stats.norm.ppf(alpha)
            var_1d = -(mu + z * sigma)

        elif method == VaRMethod.MONTE_CARLO:
            mu = self._port_returns.mean()
            sigma = self._port_returns.std()
            simulated = np.random.normal(mu, sigma, n_simulations)
            var_1d = -np.percentile(simulated, alpha * 100)

        else:
            raise ValueError(f"Unknown VaR method: {method}")

        return float(var_1d * np.sqrt(horizon))

    # ------------------------------------------------------------------
    # Conditional Value at Risk (CVaR / Expected Shortfall)
    # ------------------------------------------------------------------

    def cvar(
        self,
        confidence: float = 0.95,
        method: Union[VaRMethod, str] = VaRMethod.HISTORICAL,
        horizon: int = 1,
        n_simulations: int = 10_000,
    ) -> float:
        """
        Portfolio Conditional VaR (Expected Shortfall).

        CVaR is the expected loss beyond the VaR threshold — a coherent
        risk measure that captures tail risk better than VaR alone.

        Parameters
        ----------
        confidence : float
            Confidence level.
        method : VaRMethod or str
            ``'historical'``, ``'parametric'``, or ``'monte_carlo'``.
        horizon : int
            Holding period in days.
        n_simulations : int
            Monte Carlo paths.

        Returns
        -------
        float
            CVaR as a positive decimal.
        """
        method = VaRMethod(method)
        alpha = 1 - confidence

        if method == VaRMethod.HISTORICAL:
            threshold = np.percentile(self._port_returns, alpha * 100)
            tail = self._port_returns[self._port_returns <= threshold]
            cvar_1d = -tail.mean()

        elif method == VaRMethod.PARAMETRIC:
            mu = self._port_returns.mean()
            sigma = self._port_returns.std()
            z = stats.norm.ppf(alpha)
            cvar_1d = -(mu - sigma * stats.norm.pdf(z) / alpha)

        elif method == VaRMethod.MONTE_CARLO:
            mu = self._port_returns.mean()
            sigma = self._port_returns.std()
            simulated = np.random.normal(mu, sigma, n_simulations)
            threshold = np.percentile(simulated, alpha * 100)
            tail = simulated[simulated <= threshold]
            cvar_1d = -tail.mean()

        else:
            raise ValueError(f"Unknown method: {method}")

        return float(cvar_1d * np.sqrt(horizon))

    # ------------------------------------------------------------------
    # Drawdown analysis
    # ------------------------------------------------------------------

    def drawdown_series(self) -> pd.Series:
        """
        Compute the full drawdown series from the portfolio return stream.

        Returns
        -------
        pd.Series
            Drawdown at each date (negative values).
        """
        cumulative = (1 + self._port_returns).cumprod()
        rolling_max = cumulative.cummax()
        drawdown = (cumulative - rolling_max) / rolling_max
        drawdown.name = "Drawdown"
        return drawdown

    def max_drawdown(self) -> float:
        """
        Maximum drawdown (positive number = magnitude of worst peak-to-trough).

        Returns
        -------
        float
            Max drawdown (e.g. 0.35 = 35 % peak-to-trough loss).
        """
        dd = self.drawdown_series()
        return float(-dd.min())

    def calmar_ratio(self) -> float:
        """
        Calmar ratio: annualised return divided by maximum drawdown.

        Returns
        -------
        float
            Calmar ratio. Returns inf when max drawdown is zero.
        """
        mdd = self.max_drawdown()
        if mdd == 0:
            return np.inf
        return self.annualized_return() / mdd

    def drawdown_duration(self) -> int:
        """
        Longest drawdown duration in trading days.

        Returns
        -------
        int
            Maximum consecutive days in a drawdown.
        """
        dd = self.drawdown_series()
        in_dd = (dd < 0).astype(int)
        max_dur = 0
        current = 0
        for v in in_dd:
            if v:
                current += 1
                max_dur = max(max_dur, current)
            else:
                current = 0
        return max_dur

    # ------------------------------------------------------------------
    # Risk contribution
    # ------------------------------------------------------------------

    def risk_contribution(self) -> pd.Series:
        """
        Marginal risk contribution of each asset as a fraction of total
        portfolio volatility.

        Uses the Euler decomposition:
            RC_i = w_i * (Σw)_i / σ_p

        Returns
        -------
        pd.Series
            Risk contribution per asset (sums to 1).
        """
        cov = self.returns.cov().values * self.trading_days
        port_vol = self.annualized_volatility()
        if port_vol == 0:
            return pd.Series(
                np.ones(len(self.weights)) / len(self.weights),
                index=self.returns.columns,
                name="Risk Contribution",
            )
        sigma_w = cov @ self.weights
        rc = self.weights * sigma_w / port_vol
        # Normalise to sum to 1
        rc = rc / rc.sum()
        return pd.Series(rc, index=self.returns.columns, name="Risk Contribution")

    def marginal_risk_contribution(self) -> pd.Series:
        """
        Marginal contribution to portfolio volatility per unit weight.

        Returns
        -------
        pd.Series
            Marginal risk contribution per asset.
        """
        cov = self.returns.cov().values * self.trading_days
        port_vol = self.annualized_volatility()
        if port_vol == 0:
            return pd.Series(np.zeros(len(self.weights)), index=self.returns.columns)
        mrc = (cov @ self.weights) / port_vol
        return pd.Series(mrc, index=self.returns.columns, name="Marginal RC")

    # ------------------------------------------------------------------
    # Stress testing
    # ------------------------------------------------------------------

    def stress_test(
        self, scenarios: Optional[Dict[str, pd.Series]] = None
    ) -> pd.DataFrame:
        """
        Apply named stress scenarios to the portfolio and report P&L impact.

        Parameters
        ----------
        scenarios : dict of {str: pd.Series}
            Each value is a Series of asset-level shocks (indexed by ticker).
            If None, a built-in set of standard shocks is used.

        Returns
        -------
        pd.DataFrame
            Scenario name → portfolio return under that scenario.
        """
        if scenarios is None:
            scenarios = self._default_scenarios()

        results = {}
        tickers = self.returns.columns.tolist()
        for name, shocks in scenarios.items():
            # Align shocks to our asset universe
            shock_vec = np.array([shocks.get(t, 0.0) for t in tickers])
            pnl = float(self.weights @ shock_vec)
            results[name] = {"Scenario P&L": pnl, "Scenario P&L (%)": pnl * 100}

        return pd.DataFrame(results).T

    def _default_scenarios(self) -> Dict[str, pd.Series]:
        """Standard stress scenarios (approximate historical analogues)."""
        tickers = self.returns.columns.tolist()
        # Flat shocks applied uniformly across all assets
        scenarios = {
            "2008 GFC (-40%)": pd.Series(-0.40, index=tickers),
            "2020 COVID Crash (-30%)": pd.Series(-0.30, index=tickers),
            "2022 Rate Shock (-20%)": pd.Series(-0.20, index=tickers),
            "Equity +10% Rally": pd.Series(0.10, index=tickers),
            "Flash Crash (-10%)": pd.Series(-0.10, index=tickers),
        }
        return scenarios

    # ------------------------------------------------------------------
    # Full risk report
    # ------------------------------------------------------------------

    def full_report(self, risk_free_rate: float = 0.02) -> Dict:
        """
        Compute and return all key risk metrics as a dictionary.

        Parameters
        ----------
        risk_free_rate : float
            Annualised risk-free rate.

        Returns
        -------
        dict
            Comprehensive risk report.
        """
        report = {
            "Annualized Return": self.annualized_return(),
            "Annualized Volatility": self.annualized_volatility(),
            "Sharpe Ratio": self.sharpe_ratio(risk_free_rate),
            "Sortino Ratio": self.sortino_ratio(risk_free_rate),
            "Calmar Ratio": self.calmar_ratio(),
            "Max Drawdown": self.max_drawdown(),
            "Drawdown Duration (days)": self.drawdown_duration(),
            "VaR 95% (1-day, hist)": self.var(0.95, VaRMethod.HISTORICAL),
            "VaR 99% (1-day, hist)": self.var(0.99, VaRMethod.HISTORICAL),
            "CVaR 95% (1-day, hist)": self.cvar(0.95, VaRMethod.HISTORICAL),
            "CVaR 99% (1-day, hist)": self.cvar(0.99, VaRMethod.HISTORICAL),
        }
        logger.debug("Full risk report generated.")
        return report
