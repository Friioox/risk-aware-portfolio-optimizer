"""
data_loader.py
==============
Handles loading and preprocessing of historical price data from multiple sources:
- CSV files
- pandas DataFrames
- yfinance (live market data)

All data is normalized to a common format: a DataFrame of adjusted closing prices
indexed by date, with asset tickers as column headers.
"""

import logging
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DataLoader:
    """
    Loads and validates historical price data from various sources.

    Attributes
    ----------
    tickers : list of str
        Asset tickers to load.
    start_date : str
        Start date in 'YYYY-MM-DD' format.
    end_date : str
        End date in 'YYYY-MM-DD' format.
    prices : pd.DataFrame
        Adjusted closing prices indexed by date.
    """

    def __init__(
        self,
        tickers: Optional[List[str]] = None,
        start_date: str = "2018-01-01",
        end_date: str = "2024-01-01",
    ):
        self.tickers = tickers or []
        self.start_date = start_date
        self.end_date = end_date
        self.prices: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def from_yfinance(self, tickers: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Download adjusted closing prices via yfinance.

        Parameters
        ----------
        tickers : list of str, optional
            Override instance tickers.

        Returns
        -------
        pd.DataFrame
            Price DataFrame indexed by date.
        """
        try:
            import yfinance as yf
        except ImportError as exc:
            raise ImportError(
                "yfinance is required: pip install yfinance"
            ) from exc

        tickers = tickers or self.tickers
        if not tickers:
            raise ValueError("No tickers provided.")

        logger.info("Downloading data for %s from yfinance …", tickers)
        raw = yf.download(
            tickers,
            start=self.start_date,
            end=self.end_date,
            auto_adjust=True,
            progress=False,
        )

        # yfinance returns multi-level columns when >1 ticker
        if isinstance(raw.columns, pd.MultiIndex):
            prices = raw["Close"]
        else:
            prices = raw[["Close"]]
            prices.columns = tickers

        prices = self._clean(prices)
        self.prices = prices
        logger.info("Downloaded %d rows × %d assets.", *prices.shape)
        return prices

    def from_csv(
        self,
        path: Union[str, Path],
        date_column: str = "Date",
        price_columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Load prices from a CSV file.

        The CSV must contain a date column and one price column per asset.

        Parameters
        ----------
        path : str or Path
            Path to the CSV file.
        date_column : str
            Name of the date column.
        price_columns : list of str, optional
            Columns to use as assets. Defaults to all non-date columns.

        Returns
        -------
        pd.DataFrame
            Price DataFrame indexed by date.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"CSV not found: {path}")

        logger.info("Loading CSV: %s", path)
        df = pd.read_csv(path, parse_dates=[date_column], index_col=date_column)

        if price_columns:
            df = df[price_columns]

        prices = self._clean(df)
        self.prices = prices
        logger.info("Loaded %d rows × %d assets from CSV.", *prices.shape)
        return prices

    def from_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Accept an externally constructed price DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Price DataFrame; index must be datetime-like.

        Returns
        -------
        pd.DataFrame
            Cleaned price DataFrame.
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        prices = self._clean(df)
        self.prices = prices
        logger.info("Accepted DataFrame: %d rows × %d assets.", *prices.shape)
        return prices

    def get_prices(self) -> pd.DataFrame:
        """Return loaded prices, raising if none available."""
        if self.prices is None:
            raise RuntimeError("No price data loaded. Call from_*() first.")
        return self.prices

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _clean(df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize a price DataFrame:
        - ensure DatetimeIndex sorted ascending
        - drop columns that are entirely NaN
        - forward-fill minor gaps then back-fill any leading NaN
        - cast to float64
        """
        df = df.copy()
        df.index = pd.to_datetime(df.index)
        df.sort_index(inplace=True)

        # Drop assets with no data at all
        df.dropna(axis=1, how="all", inplace=True)

        # Fill small gaps
        df.ffill(inplace=True)
        df.bfill(inplace=True)

        df = df.astype(np.float64)

        if df.isnull().values.any():
            logger.warning("Remaining NaN values after cleaning. Check input data.")

        return df

    def describe(self) -> pd.DataFrame:
        """Return summary statistics for the loaded price series."""
        prices = self.get_prices()
        return prices.describe()

    def trim_to_common_history(self) -> pd.DataFrame:
        """
        Drop rows where any asset has missing data to ensure a balanced panel.

        Returns
        -------
        pd.DataFrame
            Trimmed price DataFrame with no missing values.
        """
        prices = self.get_prices()
        trimmed = prices.dropna()
        dropped = len(prices) - len(trimmed)
        if dropped:
            logger.info("Dropped %d rows to align price history.", dropped)
        self.prices = trimmed
        return trimmed
