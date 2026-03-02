# strategy/strategyV02.py
"""
Blank-slate strategy: no entry conditions. Signal and entry_mode are set to 0 and ""
for all bars. Entry conditions are to be added later when implementing the new strategy.
"""

import pandas as pd


class HiLoATRBands:
    """Blank-slate strategy: no signals generated. Entry conditions to be added later."""

    def __init__(self, config=None):
        """Read config for compatibility with backtester/WFO; no entry-condition params used."""
        if config:
            _ = config.get("strategy", {}).get("params", {})
        print("Strategy HiLoATRBands: blank slate (no entry signals)")

    def generate_signals(self, df):
        """
        Blank slate: set signal=0 and entry_mode="" for all bars. No entry conditions.
        Only requires standard OHLC + index.
        """
        if df is None or len(df) == 0:
            raise ValueError("DataFrame is empty or None")

        required = ["low", "high", "close"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        df = df.copy()
        df["signal"] = 0
        df["entry_mode"] = ""
        return df
