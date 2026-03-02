# indicators/st_trend_oscillator_pro.py
"""
ST_TrendOscillatorPRO_X - Simpler Trading Trend Oscillator (ThinkScript port).
Uses higher timeframe close for momentum; for 5-min: 30-min HTF, L1=50, L2=75.
Computed only on completed HTF bars (e.g. every 30 min), then forward-filled to
5-min bars for analysis/export; no strategy logic changes. No look-ahead.
"""
import numpy as np
import pandas as pd


def _get_preset(timeframe):
    """
    Map timeframe string to (htf_resample, L1, L2) per ThinkScript preset logic.
    Timeframe can be '5m', '5min', '1m', '15m', '1h', etc.
    """
    tf = (str(timeframe) or "5m").strip().lower()
    if "5" in tf and "min" in tf or tf == "5m":
        return "30min", 50, 75
    if "3" in tf and "min" in tf or tf == "3m":
        return "15min", 50, 80
    if "2" in tf and "min" in tf or tf == "2m":
        return "15min", 50, 65
    if "1" in tf and "min" in tf or tf == "1m":
        return "15min", 50, 50
    if "10" in tf:
        return "1h", 50, 50
    if "15" in tf:
        return "1h", 50, 65
    if "30" in tf:
        return "4h", 53, 60
    if "1h" in tf or "60" in tf:
        return "1d", 34, 24
    if "2h" in tf:
        return "3d", 25, 40
    if "4h" in tf or "240" in tf:
        return "3d", 40, 80
    # Default 5-min
    return "30min", 50, 75


def compute_st_trend_oscillator_pro(
    df,
    timeframe="5m",
    trend_osc_seed=None,
    ema_seed=None,
    prev_30min_close_seed=None,
):
    """
    Compute ST_TrendOscillatorPRO_X (ThinkScript port).
    Requires 'close'. Uses higher timeframe (e.g. 30-min for 5-min bars) for momentum.
    Adds: st_trend_oscillator, st_trend_ema, st_trend_oscillator_sim, st_trend_ema_sim,
    st_trend_bullish, st_trend_bullish_cross, st_trend_bearish_cross,
    st_trend_macd_hist, st_trend_macd_bullish, st_trend_extreme, st_trend_ema_extreme.

    Optional seeds (trend_osc_seed, ema_seed) initialize the indicator from the previous
    session close for accuracy from bar 0. prev_30min_close_seed sets the first bar's
    prior close; if omitted, uses df["open"].iloc[0].
    """
    if "close" not in df.columns:
        raise ValueError("DataFrame must have 'close' column")

    htf_resample, L1, L2 = _get_preset(timeframe)
    use_seed = trend_osc_seed is not None and ema_seed is not None
    htf_prev_seed = None

    # Resample to HTF: use completed bars only (no developing bar)
    htf = df["close"].resample(htf_resample).last().dropna()
    htf_prev = htf.shift(1)

    # Compute entirely on HTF series
    diff_htf = htf - htf_prev

    if use_seed:
        # First bar: use seed for prev close so diff is defined
        htf_prev_seed = prev_30min_close_seed
        if htf_prev_seed is None and "open" in df.columns:
            htf_prev_seed = float(df["open"].iloc[0])
        if htf_prev_seed is not None:
            diff_htf = diff_htf.copy()
            diff_htf.iloc[0] = float(htf.iloc[0]) - htf_prev_seed
        # Back out A1_init, A2_init from trend_osc_seed (A3 = osc/50 - 1)
        A3_init = trend_osc_seed / 50.0 - 1.0
        A1_init = A3_init
        A2_init = 1.0
        # Prepend init so ewm's first output = init; iloc[1:] gives correct bar 0
        diff_vals = diff_htf.values
        prepended = np.concatenate([[A1_init], diff_vals])
        A1 = pd.Series(
            pd.Series(prepended).ewm(span=L1, adjust=False).mean().values[1:],
            index=htf.index,
        )
        diff_abs_vals = np.abs(diff_vals)
        prepended_abs = np.concatenate([[A2_init], diff_abs_vals])
        A2 = pd.Series(
            pd.Series(prepended_abs).ewm(span=L1, adjust=False).mean().values[1:],
            index=htf.index,
        )
    else:
        A1 = diff_htf.ewm(span=L1, adjust=False).mean()
        A2 = diff_htf.abs().ewm(span=L1, adjust=False).mean()
    A3 = pd.Series(np.where(A2 != 0, A1 / A2, 0), index=htf.index)
    TrendOscillator_htf = 50 * (A3 + 1)
    if use_seed:
        to_vals = TrendOscillator_htf.values
        prepended_to = np.concatenate([[ema_seed], to_vals])
        ema_osc_htf = pd.Series(
            pd.Series(prepended_to).ewm(span=L2, adjust=False).mean().values[1:],
            index=htf.index,
        )
    else:
        ema_osc_htf = TrendOscillator_htf.ewm(span=L2, adjust=False).mean()

    # Map back to 5-min: ffill so each 5-min bar gets the most recent completed HTF value
    TrendOscillator = TrendOscillator_htf.reindex(df.index).ffill()
    ema_osc = ema_osc_htf.reindex(df.index).ffill()

    # Simulated values: use previous official A1, A2, EMA; plug in current 5-min close.
    # Use shift(1) so at 30-min bars we blend new diff with prior state (not current).
    A1_5m = A1.shift(1).reindex(df.index).ffill()
    A2_5m = A2.shift(1).reindex(df.index).ffill()
    ema_osc_prev = ema_osc_htf.shift(1).reindex(df.index).ffill()
    prev_30min_close = htf.shift(1).reindex(df.index).ffill()
    if use_seed:
        if htf_prev_seed is not None:
            prev_30min_close = prev_30min_close.fillna(htf_prev_seed)
        A3_init = trend_osc_seed / 50.0 - 1.0
        A1_5m = A1_5m.fillna(A3_init)
        A2_5m = A2_5m.fillna(1.0)
        ema_osc_prev = ema_osc_prev.fillna(ema_seed)

    diff_sim = df["close"] - prev_30min_close
    alpha1 = 2.0 / (L1 + 1)
    alpha2 = 2.0 / (L2 + 1)
    A1_sim = alpha1 * diff_sim + (1 - alpha1) * A1_5m
    A2_sim = alpha1 * diff_sim.abs() + (1 - alpha1) * A2_5m
    A3_sim = pd.Series(
        np.where(A2_sim != 0, A1_sim / A2_sim, 0), index=df.index
    )
    TrendOscillator_sim = 50 * (A3_sim + 1)
    ema_osc_sim = alpha2 * TrendOscillator_sim + (1 - alpha2) * ema_osc_prev

    df["st_trend_oscillator"] = TrendOscillator.round(4)
    df["st_trend_ema"] = ema_osc.round(4)
    df["st_trend_oscillator_sim"] = TrendOscillator_sim.round(4)
    df["st_trend_ema_sim"] = ema_osc_sim.round(4)
    df["st_trend_bullish"] = TrendOscillator > ema_osc

    # Cross signals: only at HTF bar boundaries (where new value appears)
    osc_above_htf = TrendOscillator_htf > ema_osc_htf
    osc_below_htf = TrendOscillator_htf <= ema_osc_htf
    prev_above_htf = osc_above_htf.shift(1).fillna(False)
    prev_below_htf = osc_below_htf.shift(1).fillna(False)
    bullish_cross_htf = osc_above_htf & prev_below_htf
    bearish_cross_htf = osc_below_htf & prev_above_htf
    df["st_trend_bullish_cross"] = bullish_cross_htf.reindex(df.index).fillna(False)
    df["st_trend_bearish_cross"] = bearish_cross_htf.reindex(df.index).fillna(False)

    # MACD (21, 55, 55) on close
    close = df["close"]
    macd_line = close.ewm(span=21, adjust=False).mean() - close.ewm(span=55, adjust=False).mean()
    signal_line = macd_line.ewm(span=55, adjust=False).mean()
    macd_hist = macd_line - signal_line
    df["st_trend_macd_hist"] = macd_hist.round(4)
    df["st_trend_macd_bullish"] = macd_hist > 0

    # BullCycle2 / BearCycle2
    df["st_trend_extreme"] = (TrendOscillator > 69) | (TrendOscillator < 31)
    # BullCycle3 / BearCycle3
    df["st_trend_ema_extreme"] = (ema_osc > 69) | (ema_osc < 31)

    return df
