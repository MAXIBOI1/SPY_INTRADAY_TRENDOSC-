# indicators/tmo.py
"""TMO - True Momentum Oscillator. Requires open, close. Adds tmo_main, tmo_signal."""
# TMO vs TOS: Formula matches Thinkorswim ExpAverage and fold logic. For values to match,
# use the same session and 5m bars (no higher aggregation on TOS).
import numpy as np
import pandas as pd


def compute_tmo(df, length=14, calc_length=5, smooth_length=3):
    """
    Compute TMO (True Momentum Oscillator).
    Requires 'open' and 'close'. Adds columns: tmo_main, tmo_signal,
    tmo_overbought, tmo_oversold (constant levels for reference),
    tmo_bullish (True when main > signal, i.e. TOS green), and
    tmo_bullish_bearish ("Bullish"/"Bearish") for readability.
    """
    required_cols = ["open", "close"]
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"DataFrame must have columns: {required_cols}")

    # Raw momentum: for each bar, sum over j in 0..length-1 of
    # +1 if close > open[j bars ago], -1 if close < open[j bars ago], else 0
    data = pd.Series(0.0, index=df.index, dtype=float)
    for j in range(length):
        diff = df["close"] - df["open"].shift(j)
        data += np.sign(diff).fillna(0)

    # EWM chain: data -> EMA(calc_length) -> Main -> EMA(smooth_length) -> Signal
    ema5 = data.ewm(span=calc_length, adjust=False).mean()
    main = ema5.ewm(span=smooth_length, adjust=False).mean()
    signal = main.ewm(span=smooth_length, adjust=False).mean()

    # Optional: round for CSV display (cosmetic; matches TOS chart display)
    df["tmo_main"] = main.round(4)
    df["tmo_signal"] = signal.round(4)

    # Bullish = main > signal (TOS green); otherwise bearish (TOS red)
    df["tmo_bullish"] = main > signal
    df["tmo_bullish_bearish"] = np.where(main > signal, "Bullish", "Bearish")

    ob = round(length * 0.8)
    df["tmo_overbought"] = ob
    df["tmo_oversold"] = -ob

    return df
