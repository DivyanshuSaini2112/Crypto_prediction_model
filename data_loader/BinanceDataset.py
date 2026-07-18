"""
BinanceDataset.py — Live crypto data loader using Yahoo Finance (yfinance).

Why Yahoo Finance instead of BitMEX / Binance:
  - Works from India (BitMEX geo-blocked, Binance/CoinGecko/Bybit all blocked)
  - Completely FREE — no API key needed
  - Covers all 18 project coins
  - Daily data back to 2014 for BTC; hourly data available for last 730 days

Symbols: Uses the same project symbol names (XBTUSD, ETHUSD, etc.)
         automatically mapped to Yahoo Finance tickers (BTC-USD, ETH-USD, etc.)

Usage:
    python train.py dataset_loader=Binance model=xgboost
    python train.py dataset_loader=Binance dataset_loader.symbol=ETHUSD model=random_forest
    python train.py dataset_loader=Binance dataset_loader.binsize=1h model=lstm
"""

import logging
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime

from .creator import create_dataset, preprocess

logger = logging.getLogger(__name__)

# ── Symbol map: project symbols → Yahoo Finance tickers ──────────────────────
SYMBOL_MAP = {
    'XBTUSD':  'BTC-USD',
    'ETHUSD':  'ETH-USD',
    'BNBUSD':  'BNB-USD',
    'ADAUSD':  'ADA-USD',
    'DOGEUSD': 'DOGE-USD',
    'SOLUSD':  'SOL-USD',
    'DOTUSD':  'DOT-USD',
    'LTCUSD':  'LTC-USD',
    'TRXUSD':  'TRX-USD',
    'AVAXUSD': 'AVAX-USD',
    'LINKUSD': 'LINK-USD',
    'NEARUSD': 'NEAR-USD',
    'BCHUSD':  'BCH-USD',
    'AXSUSD':  'AXS-USD',
    'EOSUSD':  'EOS-USD',
    'APEUSD':  'APE-USD',
    'APTUSD':  'APT-USD',
    'CROUSD':  'CRO-USD',
}

# ── Interval map: project binsize → yfinance interval string ─────────────────
INTERVAL_MAP = {
    '1d': '1d',
    '1h': '1h',
    '5m': '5m',
    '1m': '1m',
}

# yfinance API limits per interval (use period= fallback if dates out of range)
MAX_PERIOD = {
    '1m':  '7d',
    '5m':  '60d',
    '1h':  '730d',
    '1d':  'max',   # unlimited for daily
}


class BinanceDataset:
    """
    Live crypto data loader using Yahoo Finance (yfinance).
    No API key required — works globally including India.

    Configured via: dataset_loader=Binance in Hydra.
    """

    def __init__(self, cfg):
        self.cfg       = cfg
        args           = cfg.dataset_loader
        raw_sym        = args.symbol              # e.g. "XBTUSD"
        self.yf_symbol = SYMBOL_MAP.get(raw_sym, raw_sym.replace('USD', '-USD'))
        self.interval  = INTERVAL_MAP.get(getattr(args, 'binsize', '1d'), '1d')
        self.window_size = args.window_size

        logger.info(
            f"[YahooFinance Loader] {raw_sym} -> {self.yf_symbol} "
            f"| interval={self.interval}"
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _fetch(self, start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
        """Download OHLCV data via yfinance."""
        logger.info(
            f"Downloading {self.yf_symbol} ({self.interval}) from Yahoo Finance "
            f"[{start_dt.date()} -> {end_dt.date()}] ..."
        )
        ticker = yf.Ticker(self.yf_symbol)

        # Try date-range download first; fall back to period= for sub-daily intervals
        try:
            df = ticker.history(
                start=start_dt.strftime('%Y-%m-%d'),
                end=end_dt.strftime('%Y-%m-%d'),
                interval=self.interval,
                auto_adjust=True,
            )
        except Exception as e:
            period = MAX_PERIOD.get(self.interval, 'max')
            logger.warning(f"date-range download failed ({e}), falling back to period={period}")
            df = ticker.history(period=period, interval=self.interval, auto_adjust=True)

        if df is None or df.empty:
            raise RuntimeError(
                f"No data returned for {self.yf_symbol}. "
                f"Check symbol or try a different date range."
            )

        logger.info(f"Downloaded {len(df)} candles for {self.yf_symbol}")
        return df

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalise yfinance DataFrame columns to match the project convention."""
        df = df.reset_index()

        # yfinance returns 'Datetime' for intra-day, 'Date' for daily
        date_col = 'Datetime' if 'Datetime' in df.columns else 'Date'
        df = df.rename(columns={date_col: 'Date'})

        # yfinance returns Title-Case; project expects mixed (open/close/volume lowercase, High/Low capitalised)
        df = df.rename(columns={
            'Open':   'open',
            'Close':  'close',
            'Volume': 'volume',
        })

        keep = ['Date', 'open', 'High', 'Low', 'close', 'volume']
        df = df[[c for c in keep if c in df.columns]]

        # Ensure Date is timezone-aware UTC
        if df['Date'].dt.tz is None:
            df['Date'] = df['Date'].dt.tz_localize('UTC')

        df = df.dropna().reset_index(drop=True)
        return df

    # ── Public interface ──────────────────────────────────────────────────────

    def get_dataset(self):
        args     = self.cfg.dataset_loader
        start_dt = datetime.strptime(str(args.train_start_date), '%Y-%m-%d %H:%M:%S')
        end_dt   = datetime.strptime(str(args.valid_end_date),   '%Y-%m-%d %H:%M:%S')

        raw = self._fetch(start_dt, end_dt)
        df  = self._clean(raw)

        logger.info(
            f"Final dataset: {len(df)} rows | "
            f"{df['Date'].iloc[0].date()} -> {df['Date'].iloc[-1].date()}"
        )

        dataset, profit_calculator = preprocess(df, self.cfg, logger)
        return dataset, profit_calculator
