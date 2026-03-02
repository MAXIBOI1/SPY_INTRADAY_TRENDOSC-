# strategy/strategyV02.py
"""
HiLo + ATR Bands strategy.
Long: after bar with hilopro_slow_d >= 80 and price touches atr_bands_lower, enter on first bar (within N bars) where close > open and hilopro_slow_d >= 80; else void.
Short: after bar with hilopro_slow_d <= 20 and price touches atr_bands_upper, enter on first bar (within N bars) where close < open and hilopro_slow_d <= 20; else void.
Entry price is set by exit module (ATRBandsExit): close of the confirming bar; stop/target/breakeven from that bar's bands.
"""

import numpy as np
import pandas as pd


class HiLoATRBands:
    """HiLo stochastic + ATR bands: enter on first confirming candle (close vs open) within N bars of setup; void if none."""

    def __init__(self, config=None):
        params = config.get("strategy", {}).get("params", {}) if config else {}
        self.allow_short = config.get("strategy", {}).get("allow_short", True) if config else True
        warmup = params.get("indicator_warmup_bars", 0)
        self.indicator_warmup_bars = max(0, int(warmup)) if warmup is not None else 0
        self.hilopro_long_threshold = params.get("hilopro_long_threshold", 80)
        self.hilopro_short_threshold = params.get("hilopro_short_threshold", 20)
        self.close_agree_bars = int(params.get("close_agree_bars", 10))
        print(
            f"Strategy HiLoATRBands: allow_short={self.allow_short}, "
            f"indicator_warmup_bars={self.indicator_warmup_bars}, "
            f"long_threshold={self.hilopro_long_threshold}, short_threshold={self.hilopro_short_threshold}, "
            f"close_agree_bars={self.close_agree_bars}"
        )
        print("Entry mode: CLOSE-AGREE (enter on first bar with close>open for long, close<open for short within N bars; void if none)")

    def generate_signals(self, df):
        """
        Generate signals: for each long/short setup bar, look forward up to close_agree_bars for first
        confirming bar (long: close > open and hilopro_slow_d >= threshold; short: close < open and
        hilopro_slow_d <= threshold). Set signal and entry_mode only on that bar; void if none found.
        """
        if df is None or len(df) == 0:
            raise ValueError("DataFrame is empty or None")

        required = [
            "hilopro_slow_d", "atr_bands_lower", "atr_bands_center", "atr_bands_upper",
            "low", "high", "open", "close",
        ]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        df = df.copy()
        df["signal"] = 0
        df["entry_mode"] = ""

        warmup = self.indicator_warmup_bars
        long_th = self.hilopro_long_threshold
        short_th = self.hilopro_short_threshold
        n_bars = self.close_agree_bars
        long_setups = 0
        long_voided = 0
        short_setups = 0
        short_voided = 0

        i = warmup
        while i < len(df):
            slow_d = df["hilopro_slow_d"].iloc[i]
            lower = float(df["atr_bands_lower"].iloc[i])
            upper = float(df["atr_bands_upper"].iloc[i])
            low_i = float(df["low"].iloc[i])
            high_i = float(df["high"].iloc[i])

            if np.isnan(slow_d) or np.isnan(lower) or np.isnan(upper):
                i += 1
                continue

            signal_val = 0
            if slow_d >= long_th and low_i <= lower:
                long_setups += 1
                # Long setup: look for first bar with close > open and hilopro_slow_d >= long_th
                for j in range(i + 1, min(i + n_bars + 1, len(df))):
                    c_j = float(df["close"].iloc[j])
                    o_j = float(df["open"].iloc[j])
                    slow_j = df["hilopro_slow_d"].iloc[j]
                    if c_j > o_j and not np.isnan(slow_j) and slow_j >= long_th:
                        df.at[df.index[j], "signal"] = 1
                        df.at[df.index[j], "entry_mode"] = "bands"
                        i = j
                        signal_val = 1
                        break
                if signal_val == 0:
                    long_voided += 1
            elif self.allow_short and slow_d <= short_th and high_i >= upper:
                short_setups += 1
                # Short setup: look for first bar with close < open and hilopro_slow_d <= short_th
                for j in range(i + 1, min(i + n_bars + 1, len(df))):
                    c_j = float(df["close"].iloc[j])
                    o_j = float(df["open"].iloc[j])
                    slow_j = df["hilopro_slow_d"].iloc[j]
                    if c_j < o_j and not np.isnan(slow_j) and slow_j <= short_th:
                        df.at[df.index[j], "signal"] = -1
                        df.at[df.index[j], "entry_mode"] = "bands"
                        i = j
                        signal_val = -1
                        break
                if signal_val == 0:
                    short_voided += 1

            if signal_val == 0:
                i += 1
            else:
                i += 1  # advance past the entry bar for next iteration

        n_long = (df["signal"] == 1).sum()
        n_short = (df["signal"] == -1).sum()
        print(f"Generated {n_long} long and {n_short} short signals (HiLoATRBands)")
        print(f"Close-agree: long setups={long_setups} (voided={long_voided}), short setups={short_setups} (voided={short_voided})")
        return df
