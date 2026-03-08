"""
optimizer.py
============
Portfolio optimization engine supporting five core strategies plus two
advanced models:

Core strategies
---------------
1. Mean-Variance Optimization  (Markowitz)
2. Mean-CVaR Optimization
3. Risk Parity
4. Minimum Volatility
5. Maximum Sharpe Ratio

Advanced models
---------------
6. Black-Litterman
7. Hierarchical Risk Parity (HRP)

All solvers return an OptimizationResult dataclass that carries weights
and key portfolio statistics so callers don't need to recompute them.
"""

import logging
import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.optimize import minimize
from scipy.spatial.distance import squareform

logger = logging.getLogger(__name__)

TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class OptimizationResult:
    """Carries all outputs of a portfolio optimization run."""

    strategy: str
    weights: np.ndarray
    asset_names: List[str]
    expected_return: float = 0.0
    volatility: float = 0.0
    sharpe_ratio: float = 0.0
    var_95: float = 0.0
    cvar_95: float = 0.0
    max_drawdown: float = 0.0
    extra: Dict = field(default_factory=dict)

    def weights_series(self) -> pd.Series:
        """Return weights as a labelled pd.Series."""
        return pd.Series(self.weights, index=self.asset_names, name="Weight")

    def summary(self) -> pd.Series:
        """One-line summary of portfolio metrics."""
        return pd.Series(
            {
                "Strategy": self.strategy,
                "Expected Return": f"{self.expected_return:.2%}",
                "Volatility": f"{self.volatility:.2%}",
                "Sharpe Ratio": f"{self.sharpe_ratio:.3f}",
                "VaR 95%": f"{self.var_95:.2%}",
                "CVaR 95%": f"{self.cvar_95:.2%}",
                "Max Drawdown": f"{self.max_drawdown:.2%}",
            }
        )


# ---------------------------------------------------------------------------
# Strategy enum
# ---------------------------------------------------------------------------

class OptimizationStrategy(str, Enum):
    MEAN_VARIANCE = "mean_variance"
    MEAN_CVAR = "mean_cvar"
    RISK_PARITY = "risk_parity"
    MIN_VOLATILITY = "min_volatility"
    MAX_SHARPE = "max_sharpe"
    BLACK_LITTERMAN = "black_litterman"
    HRP = "hrp"


# ---------------------------------------------------------------------------
# Core optimizer
# ---------------------------------------------------------------------------

class PortfolioOptimizer:
    """
    Runs portfolio optimization given return statistics and constraints.

    Parameters
    ----------
    mu : pd.Series
        Annualised expected returns per asset.
    cov : pd.DataFrame
        Annualised covariance matrix.
    returns : pd.DataFrame
        Daily return series used for CVaR / risk metrics.
    risk_free_rate : float
        Annualised risk-free rate (default 2 %).
    trading_days : int
        Annualisation factor (default 252).
    """

    def __init__(
        self,
        mu: pd.Series,
        cov: pd.DataFrame,
        returns: pd.DataFrame,
        risk_free_rate: float = 0.02,
        trading_days: int = TRADING_DAYS,
    ):
        if mu.index.tolist() != cov.index.tolist():
            raise ValueError("mu and cov must share the same asset index.")
        self.mu = mu.values.astype(float)
        self.cov = cov.values.astype(float)
        self.returns = returns.copy()
        self.asset_names = list(mu.index)
        self.n = len(self.asset_names)
        self.rf = risk_free_rate
        self.trading_days = trading_days

    # ------------------------------------------------------------------
    # Public dispatch
    # ------------------------------------------------------------------

    def optimize(
        self,
        strategy: OptimizationStrategy = OptimizationStrategy.MAX_SHARPE,
        constraints: Optional[Dict] = None,
        **kwargs,
    ) -> OptimizationResult:
        """
        Run optimization for the requested strategy.

        Parameters
        ----------
        strategy : OptimizationStrategy
            Which optimization strategy to use.
        constraints : dict, optional
            Constraint dictionary produced by ``constraints.py``.
            Expected keys (all optional):
                ``bounds``              – list of (min, max) per asset
                ``long_only``           – bool
                ``max_weight``          – float
                ``min_weight``          – float
                ``target_volatility``   – float
                ``turnover_limit``      – float
                ``current_weights``     – np.ndarray
        **kwargs
            Strategy-specific keyword arguments forwarded to the solver.

        Returns
        -------
        OptimizationResult
        """
        constraints = constraints or {}
        strategy = OptimizationStrategy(strategy)

        dispatch = {
            OptimizationStrategy.MEAN_VARIANCE: self._mean_variance,
            OptimizationStrategy.MEAN_CVAR: self._mean_cvar,
            OptimizationStrategy.RISK_PARITY: self._risk_parity,
            OptimizationStrategy.MIN_VOLATILITY: self._min_volatility,
            OptimizationStrategy.MAX_SHARPE: self._max_sharpe,
            OptimizationStrategy.BLACK_LITTERMAN: self._black_litterman,
            OptimizationStrategy.HRP: self._hrp,
        }

        solver_fn = dispatch[strategy]
        weights = solver_fn(constraints, **kwargs)
        weights = self._post_process(weights, constraints)

        result = self._build_result(strategy.value, weights)
        logger.info(
            "[%s] Return=%.2f%% Vol=%.2f%% Sharpe=%.3f",
            strategy.value,
            result.expected_return * 100,
            result.volatility * 100,
            result.sharpe_ratio,
        )
        return result

    # ------------------------------------------------------------------
    # Strategy implementations
    # ------------------------------------------------------------------

    def _mean_variance(self, constraints: Dict, target_return: Optional[float] = None, **_) -> np.ndarray:
        """
        Markowitz mean-variance optimization.

        Minimizes portfolio variance subject to a target return (if provided)
        or returns the global minimum variance portfolio otherwise.
        """
        bounds = self._get_bounds(constraints)
        cons = self._scipy_constraints(constraints)

        if target_return is not None:
            cons = cons + [{
                "type": "eq",
                "fun": lambda w, tr=target_return: w @ self.mu - tr,
            }]

        def objective(w):
            return w @ self.cov @ w

        def grad(w):
            return 2 * self.cov @ w

        w0 = self._initial_weights()
        result = minimize(
            objective,
            w0,
            jac=grad,
            method="SLSQP",
            bounds=bounds,
            constraints=cons,
            options={"maxiter": 1000, "ftol": 1e-12},
        )
        return self._check_result(result, "Mean-Variance")

    def _mean_cvar(
        self,
        constraints: Dict,
        confidence: float = 0.95,
        target_return: Optional[float] = None,
        **_,
    ) -> np.ndarray:
        """
        Mean-CVaR optimization via linear programming formulation.

        Minimizes CVaR at the specified confidence level subject to an optional
        target return constraint.
        """
        try:
            import cvxpy as cp
        except ImportError:
            logger.warning("cvxpy not installed; falling back to scipy Mean-CVaR.")
            return self._mean_cvar_scipy(constraints, confidence, target_return)

        T = len(self.returns)
        alpha = 1 - confidence
        rets = self.returns.values  # T × n

        w = cp.Variable(self.n)
        z = cp.Variable(T)
        gamma = cp.Variable()

        port_ret = rets @ w  # T-vector of portfolio returns
        # CVaR = gamma + (1/alpha/T) * sum(max(-port_ret - gamma, 0))
        cvar = gamma + (1 / (alpha * T)) * cp.sum(z)

        cvx_cons = [
            cp.sum(w) == 1,
            z >= 0,
            z >= -port_ret - gamma,
        ]

        bounds = self._get_bounds(constraints)
        for i, (lo, hi) in enumerate(bounds):
            if lo is not None:
                cvx_cons.append(w[i] >= lo)
            if hi is not None:
                cvx_cons.append(w[i] <= hi)

        if target_return is not None:
            cvx_cons.append(self.mu @ w >= target_return)

        prob = cp.Problem(cp.Minimize(cvar), cvx_cons)
        prob.solve(solver=cp.ECOS, verbose=False)

        if prob.status not in ("optimal", "optimal_inaccurate"):
            logger.warning("Mean-CVaR (cvxpy) status: %s. Falling back.", prob.status)
            return self._mean_cvar_scipy(constraints, confidence, target_return)

        return np.clip(w.value, 0 if constraints.get("long_only", True) else None, None)

    def _mean_cvar_scipy(
        self,
        constraints: Dict,
        confidence: float = 0.95,
        target_return: Optional[float] = None,
    ) -> np.ndarray:
        """Scipy fallback for Mean-CVaR when cvxpy is unavailable."""
        alpha = 1 - confidence

        def objective(w):
            port = self.returns.values @ w
            threshold = np.percentile(port, alpha * 100)
            tail = port[port <= threshold]
            return -tail.mean() if len(tail) > 0 else 0.0

        bounds = self._get_bounds(constraints)
        cons = self._scipy_constraints(constraints)
        if target_return is not None:
            cons = cons + [{"type": "ineq", "fun": lambda w, tr=target_return: w @ self.mu - tr}]

        result = minimize(
            objective,
            self._initial_weights(),
            method="SLSQP",
            bounds=bounds,
            constraints=cons,
            options={"maxiter": 2000, "ftol": 1e-10},
        )
        return self._check_result(result, "Mean-CVaR (scipy)")

    def _risk_parity(self, constraints: Dict, **_) -> np.ndarray:
        """
        Risk Parity: allocate so each asset contributes equally to
        total portfolio volatility.
        """
        n = self.n
        target_rc = np.ones(n) / n  # equal contribution

        def objective(w):
            port_var = w @ self.cov @ w
            sigma_w = self.cov @ w
            rc = w * sigma_w / np.sqrt(max(port_var, 1e-12))
            rc_norm = rc / rc.sum()
            return np.sum((rc_norm - target_rc) ** 2)

        bounds = self._get_bounds(constraints)
        # Risk parity needs weights > 0; ensure lower bound >= 1e-4
        bounds = [(max(lo, 1e-4) if lo is not None else 1e-4, hi) for lo, hi in bounds]
        cons = self._scipy_constraints(constraints)

        w0 = self._initial_weights()
        result = minimize(
            objective,
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=cons,
            options={"maxiter": 2000, "ftol": 1e-14},
        )
        return self._check_result(result, "Risk Parity")

    def _min_volatility(self, constraints: Dict, **_) -> np.ndarray:
        """Minimum Variance portfolio."""
        return self._mean_variance(constraints, target_return=None)

    def _max_sharpe(self, constraints: Dict, **_) -> np.ndarray:
        """
        Maximum Sharpe Ratio via negative Sharpe minimization.
        Uses the Sharpe-maximising tangency portfolio.
        """
        bounds = self._get_bounds(constraints)
        cons = self._scipy_constraints(constraints)

        def neg_sharpe(w):
            ret = w @ self.mu
            vol = np.sqrt(max(w @ self.cov @ w, 1e-12))
            return -(ret - self.rf) / vol

        def neg_sharpe_grad(w):
            ret = w @ self.mu
            vol = np.sqrt(max(w @ self.cov @ w, 1e-12))
            d_ret = self.mu
            d_vol = (self.cov @ w) / vol
            sharpe = (ret - self.rf) / vol
            return -(d_ret / vol - sharpe * d_vol / vol)

        best_result = None
        best_val = np.inf
        for _ in range(5):  # Multiple restarts for robustness
            w0 = np.random.dirichlet(np.ones(self.n))
            res = minimize(
                neg_sharpe,
                w0,
                jac=neg_sharpe_grad,
                method="SLSQP",
                bounds=bounds,
                constraints=cons,
                options={"maxiter": 1000, "ftol": 1e-12},
            )
            if res.success and res.fun < best_val:
                best_val = res.fun
                best_result = res

        if best_result is None:
            logger.warning("Max Sharpe did not converge; using Min Variance.")
            return self._min_volatility(constraints)

        return self._check_result(best_result, "Max Sharpe")

    def _black_litterman(
        self,
        constraints: Dict,
        market_caps: Optional[np.ndarray] = None,
        views_P: Optional[np.ndarray] = None,
        views_Q: Optional[np.ndarray] = None,
        views_omega: Optional[np.ndarray] = None,
        tau: float = 0.05,
        risk_aversion: float = 3.0,
        **_,
    ) -> np.ndarray:
        """
        Black-Litterman model.

        Blends market equilibrium returns with investor views to produce
        a posterior expected return vector, then runs mean-variance.

        Parameters
        ----------
        market_caps : np.ndarray
            Market capitalisation weights (proxy for equilibrium portfolio).
            Defaults to equal weights.
        views_P : np.ndarray (k × n)
            Pick matrix mapping views to assets.
        views_Q : np.ndarray (k,)
            View returns.
        views_omega : np.ndarray (k × k)
            View uncertainty matrix. Defaults to tau * P @ Sigma @ P'.
        tau : float
            Uncertainty in prior (default 0.05).
        risk_aversion : float
            Market risk aversion coefficient (default 3.0).
        """
        if market_caps is None:
            market_caps = np.ones(self.n) / self.n

        # Equilibrium (implied) returns
        pi = risk_aversion * self.cov @ market_caps

        if views_P is None or views_Q is None:
            # No views — use CAPM implied returns directly
            mu_bl = pd.Series(pi, index=self.asset_names)
            logger.info("Black-Litterman: no views provided; using equilibrium returns.")
        else:
            views_P = np.atleast_2d(views_P)
            views_Q = np.atleast_1d(views_Q)
            if views_omega is None:
                views_omega = np.diag(np.diag(tau * views_P @ self.cov @ views_P.T))

            # Posterior expected returns (BL formula)
            tau_sigma = tau * self.cov
            M = np.linalg.inv(
                np.linalg.inv(tau_sigma)
                + views_P.T @ np.linalg.inv(views_omega) @ views_P
            )
            mu_bl_vals = M @ (
                np.linalg.inv(tau_sigma) @ pi
                + views_P.T @ np.linalg.inv(views_omega) @ views_Q
            )
            mu_bl = pd.Series(mu_bl_vals, index=self.asset_names)
            logger.info("Black-Litterman posterior returns computed.")

        # Mean-variance with BL returns
        original_mu = self.mu.copy()
        self.mu = mu_bl.values
        weights = self._max_sharpe(constraints)
        self.mu = original_mu
        return weights

    def _hrp(self, constraints: Dict, **_) -> np.ndarray:
        """
        Hierarchical Risk Parity (López de Prado, 2016).

        Clusters assets by correlation and allocates using inverse-variance
        weighting within clusters.
        """
        corr = pd.DataFrame(self.returns.values, columns=self.asset_names).corr()
        vol = pd.Series(
            np.sqrt(np.diag(self.cov)), index=self.asset_names
        )

        # Build distance matrix and linkage
        dist = np.sqrt((1 - corr) / 2)
        dist = dist.fillna(0)
        np.fill_diagonal(dist.values, 0)
        condensed = squareform(dist.values)
        link = linkage(condensed, method="ward")

        # Quasi-diagonalisation: get sorted asset order
        sorted_idx = self._get_quasi_diag(link)
        sorted_assets = [self.asset_names[i] for i in sorted_idx]

        # Recursive bisection
        weights = pd.Series(1.0, index=sorted_assets)
        clusters = [sorted_assets]
        while clusters:
            clusters = [
                c[j:k]
                for c in clusters
                for j, k in ((0, len(c) // 2), (len(c) // 2, len(c)))
                if len(c) > 1
            ]
            for i in range(0, len(clusters), 2):
                if i + 1 >= len(clusters):
                    break
                c0, c1 = clusters[i], clusters[i + 1]
                var0 = self._cluster_var(c0, corr, vol)
                var1 = self._cluster_var(c1, corr, vol)
                alpha = 1 - var0 / (var0 + var1)
                weights[c0] *= alpha
                weights[c1] *= 1 - alpha

        # Reorder to match original asset order
        weights = weights.reindex(self.asset_names).fillna(0.0)

        # Respect long_only constraint
        if constraints.get("long_only", True):
            weights = weights.clip(lower=0)

        total = weights.sum()
        return (weights / total).values if total > 0 else np.ones(self.n) / self.n

    # ------------------------------------------------------------------
    # Efficient frontier
    # ------------------------------------------------------------------

    def efficient_frontier(
        self,
        n_points: int = 50,
        constraints: Optional[Dict] = None,
    ) -> pd.DataFrame:
        """
        Trace the mean-variance efficient frontier.

        Parameters
        ----------
        n_points : int
            Number of points on the frontier (default 50).
        constraints : dict, optional
            Shared constraint set for all frontier portfolios.

        Returns
        -------
        pd.DataFrame
            Columns: Return, Volatility, Sharpe, Weights_<asset>…
        """
        constraints = constraints or {}
        min_ret = self.mu.min()
        max_ret = self.mu.max()
        target_returns = np.linspace(min_ret * 1.01, max_ret * 0.99, n_points)

        records = []
        for tr in target_returns:
            try:
                w = self._mean_variance(constraints, target_return=tr)
                ret = float(w @ self.mu)
                vol = float(np.sqrt(w @ self.cov @ w))
                sharpe = (ret - self.rf) / vol if vol > 0 else np.nan
                rec = {"Return": ret, "Volatility": vol, "Sharpe": sharpe}
                rec.update({f"w_{a}": w[i] for i, a in enumerate(self.asset_names)})
                records.append(rec)
            except Exception as exc:
                logger.debug("Frontier point skipped: %s", exc)

        return pd.DataFrame(records)

    # ------------------------------------------------------------------
    # Monte Carlo simulation
    # ------------------------------------------------------------------

    def monte_carlo_simulation(
        self, n_portfolios: int = 5_000, seed: int = 42
    ) -> pd.DataFrame:
        """
        Simulate random portfolios and return their risk/return profiles.

        Parameters
        ----------
        n_portfolios : int
            Number of random portfolios to simulate.
        seed : int
            Random seed for reproducibility.

        Returns
        -------
        pd.DataFrame
            Columns: Return, Volatility, Sharpe, Weights_<asset>…
        """
        rng = np.random.default_rng(seed)
        records = []
        for _ in range(n_portfolios):
            w = rng.dirichlet(np.ones(self.n))
            ret = float(w @ self.mu)
            vol = float(np.sqrt(max(w @ self.cov @ w, 0)))
            sharpe = (ret - self.rf) / vol if vol > 0 else np.nan
            rec = {"Return": ret, "Volatility": vol, "Sharpe": sharpe}
            rec.update({f"w_{a}": w[i] for i, a in enumerate(self.asset_names)})
            records.append(rec)
        return pd.DataFrame(records)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _initial_weights(self) -> np.ndarray:
        """Equal-weight starting point."""
        return np.ones(self.n) / self.n

    def _get_bounds(self, constraints: Dict) -> List[Tuple]:
        """
        Build scipy bounds from constraint dict.

        Keys used:
            bounds      – explicit list of (lo, hi) tuples
            long_only   – if True, lo = 0
            min_weight  – scalar lower bound
            max_weight  – scalar upper bound
        """
        if "bounds" in constraints:
            return constraints["bounds"]

        lo = 0.0 if constraints.get("long_only", True) else -1.0
        lo = max(lo, constraints.get("min_weight", lo))
        hi = constraints.get("max_weight", 1.0)
        return [(lo, hi)] * self.n

    def _scipy_constraints(self, constraints: Dict) -> List[Dict]:
        """Build scipy equality/inequality constraints."""
        cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

        if "target_volatility" in constraints:
            tv = constraints["target_volatility"]
            cons.append({
                "type": "ineq",
                "fun": lambda w, t=tv: t - np.sqrt(max(w @ self.cov @ w, 0)),
            })

        if "turnover_limit" in constraints and "current_weights" in constraints:
            tl = constraints["turnover_limit"]
            cw = constraints["current_weights"]
            cons.append({
                "type": "ineq",
                "fun": lambda w, t=tl, c=cw: t - np.sum(np.abs(w - c)),
            })

        return cons

    @staticmethod
    def _check_result(result, name: str) -> np.ndarray:
        """Validate scipy OptimizeResult and return weights."""
        if not result.success:
            logger.warning(
                "%s optimizer did not fully converge: %s", name, result.message
            )
        w = np.clip(result.x, 0, None)
        total = w.sum()
        return w / total if total > 0 else np.ones(len(w)) / len(w)

    def _post_process(self, weights: np.ndarray, constraints: Dict) -> np.ndarray:
        """
        Final weight clean-up:
        - clip to bounds
        - renormalise
        - set tiny weights to zero
        """
        lo = 0.0 if constraints.get("long_only", True) else -1.0
        hi = constraints.get("max_weight", 1.0)
        weights = np.clip(weights, lo, hi)
        weights[np.abs(weights) < 1e-6] = 0.0
        total = np.abs(weights).sum()
        return weights / total if total > 0 else np.ones(self.n) / self.n

    def _build_result(self, strategy: str, weights: np.ndarray) -> OptimizationResult:
        """Compute and package portfolio statistics from weights."""
        from risk_metrics import RiskMetrics, VaRMethod

        ret = float(weights @ self.mu)
        vol = float(np.sqrt(max(weights @ self.cov @ weights, 0)))
        sharpe = (ret - self.rf) / vol if vol > 0 else 0.0

        # Risk metrics on daily returns
        rm = RiskMetrics(self.returns, weights, self.trading_days)
        var95 = rm.var(0.95, VaRMethod.HISTORICAL)
        cvar95 = rm.cvar(0.95, VaRMethod.HISTORICAL)
        mdd = rm.max_drawdown()

        return OptimizationResult(
            strategy=strategy,
            weights=weights,
            asset_names=self.asset_names,
            expected_return=ret,
            volatility=vol,
            sharpe_ratio=sharpe,
            var_95=var95,
            cvar_95=cvar95,
            max_drawdown=mdd,
        )

    # ------------------------------------------------------------------
    # HRP helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_quasi_diag(link: np.ndarray) -> List[int]:
        """Extract leaf ordering from linkage matrix for quasi-diagonalisation."""
        n = int(link[-1, 3])  # total number of leaves
        n_original = n  # total assets

        def recurse(node):
            if node < n_original:
                return [int(node)]
            left = int(link[int(node) - n_original, 0])
            right = int(link[int(node) - n_original, 1])
            return recurse(left) + recurse(right)

        root = 2 * n_original - 2
        return recurse(root)

    @staticmethod
    def _cluster_var(
        cluster: List[str], corr: pd.DataFrame, vol: pd.Series
    ) -> float:
        """Compute variance of the inverse-vol weighted sub-portfolio."""
        w_iv = 1.0 / vol[cluster]
        w_iv /= w_iv.sum()
        cov_cluster = corr.loc[cluster, cluster].values * np.outer(
            vol[cluster].values, vol[cluster].values
        )
        return float(w_iv.values @ cov_cluster @ w_iv.values)
