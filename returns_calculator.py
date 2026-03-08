"""
returns_calculator.py
=====================
Computes all return and risk statistics derived from a price DataFrame:

- log returns
- simple returns
- annualised expected returns
- annualised covariance matrix
- annualised correlation matrix
- annualised per-asset volatility
- rolling statistics

All methods operate on a price DataFrame (date index, asset columns)
and return consistently indexed pandas objects.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Default trading days per year used throughout the project
TRADING_DAYS = 252


class ReturnsCalculator:
    """
    Derives return and risk statistics from historical prices.

    Parameters
    ----------
    prices : pd.DataFrame
        Adjusted closing prices; rows = dates, columns = assets.
    trading_days : int
        Number of trading days per year (default 252).
    """

    def __init__(self, prices: pd.DataFrame, trading_days: int = TRADING_DAYS):
        if prices.empty:
            raise ValueError("Price DataFrame is empty.")
        self.prices = prices.copy()
        self.trading_days = trading_days

        # Lazily computed, cached after first call
        self._log_returns: Optional[pd.DataFrame] = None
        self._simple_returns: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Return series
    # ------------------------------------------------------------------

    @property
    def log_returns(self) -> pd.DataFrame:
        """Daily log returns: ln(P_t / P_{t-1})."""
        if self._log_returns is None:
            self._log_returns = np.log(self.prices / self.prices.shift(1)).dropna()
        return self._log_returns

    @property
    def simple_returns(self) -> pd.DataFrame:
        """Daily simple returns: (P_t - P_{t-1}) / P_{t-1}."""
        if self._simple_returns is None:
            self._simple_returns = self.prices.pct_change().dropna()
        return self._simple_returns

    # ------------------------------------------------------------------
    # Annualised statistics
    # ------------------------------------------------------------------

    def annualized_returns(self, use_log: bool = True) -> pd.Series:
        """
        Annualised expected return per asset.

        Parameters
        ----------
        use_log : bool
            If True use log returns (default); otherwise simple returns.

        Returns
        -------
        pd.Series
            Annualised mean return per asset.
        """
        rets = self.log_returns if use_log else self.simple_returns
        ann = rets.mean() * self.trading_days
        logger.debug("Annualised returns computed.")
        return ann

    def annualized_volatility(self, use_log: bool = True) -> pd.Series:
        """
        Annualised volatility (standard deviation) per asset.

        Parameters
        ----------
        use_log : bool
            If True use log returns (default).

        Returns
        -------
        pd.Series
            Annualised volatility per asset.
        """
        rets = self.log_returns if use_log else self.simple_returns
        vol = rets.std() * np.sqrt(self.trading_days)
        logger.debug("Annualised volatility computed.")
        return vol

    def covariance_matrix(self, use_log: bool = True) -> pd.DataFrame:
        """
        Annualised covariance matrix.

        Parameters
        ----------
        use_log : bool
            If True use log returns (default).

        Returns
        -------
        pd.DataFrame
            Annualised covariance matrix (n_assets × n_assets).
        """
        rets = self.log_returns if use_log else self.simple_returns
        cov = rets.cov() * self.trading_days
        logger.debug("Covariance matrix computed (shape %s).", cov.shape)
        return cov

    def correlation_matrix(self, use_log: bool = True) -> pd.DataFrame:
        """
        Pearson correlation matrix.

        Parameters
        ----------
        use_log : bool
            If True use log returns (default).

        Returns
        -------
        pd.DataFrame
            Correlation matrix (n_assets × n_assets).
        """
        rets = self.log_returns if use_log else self.simple_returns
        corr = rets.corr()
        logger.debug("Correlation matrix computed.")
        return corr

    # ------------------------------------------------------------------
    # Portfolio-level helpers
    # ------------------------------------------------------------------

    def portfolio_return(self, weights: np.ndarray, use_log: bool = True) -> float:
        """
        Expected annualised portfolio return given asset weights.

        Parameters
        ----------
        weights : np.ndarray
            1-D array of asset weights (must sum to 1).
        use_log : bool
            Basis for return calculation.

        Returns
        -------
        float
            Annualised portfolio return.
        """
        mu = self.annualized_returns(use_log).values
        return float(weights @ mu)

    def portfolio_volatility(self, weights: np.ndarray, use_log: bool = True) -> float:
        """
        Annualised portfolio volatility given asset weights.

        Parameters
        ----------
        weights : np.ndarray
            1-D array of asset weights.
        use_log : bool
            Basis for covariance calculation.

        Returns
        -------
        float
            Annualised portfolio standard deviation.
        """
        cov = self.covariance_matrix(use_log).values
        variance = weights @ cov @ weights
        return float(np.sqrt(max(variance, 0.0)))

    def sharpe_ratio(
        self,
        weights: np.ndarray,
        risk_free_rate: float = 0.02,
        use_log: bool = True,
    ) -> float:
        """
        Annualised Sharpe ratio.

        Parameters
        ----------
        weights : np.ndarray
            Asset weights.
        risk_free_rate : float
            Annualised risk-free rate (default 2 %).
        use_log : bool
            Basis for return / covariance calculation.

        Returns
        -------
        float
            Sharpe ratio. Returns -inf if volatility is zero.
        """
        ret = self.portfolio_return(weights, use_log)
        vol = self.portfolio_volatility(weights, use_log)
        if vol == 0:
            return -np.inf
        return (ret - risk_free_rate) / vol

    def portfolio_returns_series(
        self, weights: np.ndarray, use_log: bool = False
    ) -> pd.Series:
        """
        Daily portfolio return series given constant weights.

        Parameters
        ----------
        weights : np.ndarray
            Asset weights.
        use_log : bool
            If False (default), uses simple returns for performance tracking.

        Returns
        -------
        pd.Series
            Daily portfolio returns.
        """
        rets = self.log_returns if use_log else self.simple_returns
        port = rets @ weights
        port.name = "Portfolio"
        return port

    # ------------------------------------------------------------------
    # Rolling statistics
    # ------------------------------------------------------------------

    def rolling_volatility(
        self, weights: np.ndarray, window: int = 63, use_log: bool = True
    ) -> pd.Series:
        """
        Rolling annualised portfolio volatility.

        Parameters
        ----------
        weights : np.ndarray
            Asset weights.
        window : int
            Rolling window in trading days (default 63 ≈ 1 quarter).
        use_log : bool
            Basis for return calculation.

        Returns
        -------
        pd.Series
            Rolling portfolio volatility.
        """
        port = self.portfolio_returns_series(weights, use_log=use_log)
        rolling_vol = port.rolling(window).std() * np.sqrt(self.trading_days)
        rolling_vol.name = "Rolling Volatility"
        return rolling_vol

    def rolling_sharpe(
        self,
        weights: np.ndarray,
        window: int = 63,
        risk_free_rate: float = 0.02,
        use_log: bool = True,
    ) -> pd.Series:
        """
        Rolling annualised Sharpe ratio.

        Parameters
        ----------
        weights : np.ndarray
            Asset weights.
        window : int
            Rolling window (default 63 trading days).
        risk_free_rate : float
            Annualised risk-free rate.
        use_log : bool
            Basis for return calculation.

        Returns
        -------
        pd.Series
            Rolling Sharpe ratio.
        """
        port = self.portfolio_returns_series(weights, use_log=use_log)
        daily_rf = risk_free_rate / self.trading_days
        rolling_mean = port.rolling(window).mean()
        rolling_std = port.rolling(window).std()
        sharpe = (rolling_mean - daily_rf) / rolling_std * np.sqrt(self.trading_days)
        sharpe.name = "Rolling Sharpe"
        return sharpe

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> pd.DataFrame:
        """
        Return a summary table of annualised return and volatility per asset.

        Returns
        -------
        pd.DataFrame
            Columns: [Annualized Return, Annualized Volatility].
        """
        ret = self.annualized_returns()
        vol = self.annualized_volatility()
        sr = ret / vol  # naïve Sharpe (rf=0)
        summary = pd.DataFrame(
            {
                "Annualized Return": ret,
                "Annualized Volatility": vol,
                "Sharpe (rf=0)": sr,
            }
        )
        return summary
