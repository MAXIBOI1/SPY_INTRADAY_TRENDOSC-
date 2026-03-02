# indicators/hilopro.py
"""
ST_HiLoPRO_X (SimplerTrading) for 5-min / intraday.
Computes HiLo stochastic (SlowK, SlowD) with intraday preset, thrust state from
SlowD crosses vs OB/OS levels and OBBuy/OBSell/OSBuy/OSSell conditions.
Adds: hilopro_slow_k, hilopro_slow_d, hilopro_thrust, indicator_bullish, hilopro_arrow (↑/↓ for CSV).
"""
import numpy as np
import pandas as pd


# Intraday (5-min) preset from ST_HiLoPRO_X "else" branch
INTRADAY_A1 = 80
INTRADAY_A2 = 20
INTRADAY_P1 = 50  # K period
INTRADAY_P2 = 3   # D period
INTRADAY_P3 = 3   # slowing period

# OB/OS levels for intraday (OB1..OB15, OS1..OS15)
INTRADAY_OB = [
    47.4, 51.4, 54.9, 57.4, 62.45, 42.0, 40.0, 37.4, 35.0, 28.9, 24.0, 64.3, 67.4, 70.0, 73.9,
]
INTRADAY_OS = [
    75.5, 65.5, 63.5, 37.5, 33.1, 67.9, 40.0, 70.5, 58.6, 55.1, 60.5, 48.0, 51.6, 27.1, 45.0,
]


def _stochastic_full(high: pd.Series, low: pd.Series, close: pd.Series,
                    k_period: int, d_period: int, slowing: int) -> tuple[pd.Series, pd.Series]:
    """Full stochastic: raw %K over k_period, smooth with EWM(slowing) -> FullK, EWM(d_period) on FullK -> FullD."""
    low_min = low.rolling(window=k_period, min_periods=1).min()
    high_max = high.rolling(window=k_period, min_periods=1).max()
    denom = (high_max - low_min).replace(0, np.nan)
    raw_k = 100.0 * (close - low_min) / denom
    raw_k = raw_k.fillna(50.0)  # no range -> 50
    full_k = raw_k.ewm(span=slowing, adjust=False).mean()
    full_d = full_k.ewm(span=d_period, adjust=False).mean()
    return full_k, full_d


def _crosses_above(series: pd.Series, level: float) -> pd.Series:
    prev = series.shift(1)
    return (series > level) & (prev <= level)


def _crosses_below(series: pd.Series, level: float) -> pd.Series:
    prev = series.shift(1)
    return (series < level) & (prev >= level)


def compute_hilopro(
    df: pd.DataFrame,
    k_period: int = INTRADAY_P1,
    d_period: int = INTRADAY_P2,
    slowing: int = INTRADAY_P3,
    over_bought: float = INTRADAY_A1,
    over_sold: float = INTRADAY_A2,
) -> pd.DataFrame:
    """
    Compute ST_HiLoPRO_X for intraday (5-min). Requires high, low, close.
    Adds: hilopro_slow_k, hilopro_slow_d, hilopro_thrust, indicator_bullish, hilopro_arrow (↑/↓ for CSV).
    indicator_bullish is True when thrust == 1 (bullish), for use by the strategy.
    """
    for col in ("high", "low", "close"):
        if col not in df.columns:
            raise ValueError(f"DataFrame must have '{col}' for HiLoPRO")

    high = df["high"]
    low = df["low"]
    close = df["close"]

    full_k, full_d = _stochastic_full(high, low, close, k_period, d_period, slowing)
    df = df.copy()
    df["hilopro_slow_k"] = full_k.round(4)
    df["hilopro_slow_d"] = full_d.round(4)
    slow_d = full_d

    # OB/OS levels (intraday)
    ob = INTRADAY_OB
    os_ = INTRADAY_OS

    # upD1..upD15: SlowD crosses above OS1..OS15
    # downD1..downD15: SlowD crosses below OB1..OB15
    any_buy_cross = pd.Series(False, index=df.index)
    any_sell_cross = pd.Series(False, index=df.index)
    for i in range(15):
        any_buy_cross = any_buy_cross | _crosses_above(slow_d, os_[i])
        any_sell_cross = any_sell_cross | _crosses_below(slow_d, ob[i])

    # OBBuy, OBSell, OSBuy, OSSell (from script)
    slow_d_diff = slow_d.diff()
    OBSell = (slow_d > 80) & (slow_d < 85) & (slow_d_diff < -2.15)
    OBSell2 = (slow_d > 65) & (slow_d < 100) & (slow_d_diff < -4)
    OBSell3 = (slow_d > 69.6) & (slow_d < 100) & (slow_d_diff < -3.56)
    OBSell4 = (slow_d > 72.9) & (slow_d < 100) & (slow_d_diff < -3.14)
    OBSell5 = (slow_d > 80) & (slow_d < 86.5) & (slow_d_diff < -2.24)
    OBSell6 = (slow_d > 80) & (slow_d < 89.9) & (slow_d_diff < -2.39)
    OBBuy = (slow_d > 82.5) & (slow_d < 94.5) & (slow_d_diff > 2.13)
    OBBuy2 = (slow_d > 70) & (slow_d < 100) & (slow_d_diff > 3.75)
    OBBuy3 = (slow_d > 90.5) & (slow_d < 94.25) & (slow_d_diff > 1.5)
    OSSell = (slow_d < 23.9) & (slow_d > 12.5) & (slow_d_diff < -3)
    OSSell2 = (slow_d < 44) & (slow_d > 1) & (slow_d_diff < -5)
    OSBuy = (slow_d < 40) & (slow_d > 29.6) & (slow_d_diff > 3.5)
    OSBuy2 = (slow_d < 40) & (slow_d > 23.6) & (slow_d_diff > 4.2)
    OSBuy3 = (slow_d < 49) & (slow_d > 21.1) & (slow_d_diff > 5)
    OSBuy4 = (slow_d < 35) & (slow_d > 27.5) & (slow_d_diff > 3.2)

    any_buy = (
        any_buy_cross.fillna(False)
        | OBBuy.fillna(False) | OBBuy2.fillna(False) | OBBuy3.fillna(False)
        | OSBuy.fillna(False) | OSBuy2.fillna(False) | OSBuy3.fillna(False) | OSBuy4.fillna(False)
    )
    any_sell = (
        any_sell_cross.fillna(False)
        | OSSell.fillna(False) | OSSell2.fillna(False)
        | OBSell.fillna(False) | OBSell2.fillna(False) | OBSell3.fillna(False)
        | OBSell4.fillna(False) | OBSell5.fillna(False) | OBSell6.fillna(False)
    )

    df["hilopro_arrow"] = np.where(any_buy, "↑", np.where(any_sell, "↓", ""))

    # thrust: 1 when anyBuy, -1 when anySell, else previous (carry forward)
    thrust = pd.Series(np.nan, index=df.index, dtype=float)
    thrust.loc[any_buy] = 1.0
    thrust.loc[any_sell] = -1.0
    thrust = thrust.ffill().fillna(0.0)

    df["hilopro_thrust"] = thrust
    df["indicator_bullish"] = (thrust == 1.0)

    return df
