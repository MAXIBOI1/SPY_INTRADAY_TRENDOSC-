# indicators/atr_bands.py
"""
ATR Bands Indicator - Charles Schwab style.
Moving average center line with upper/lower bands based on ATR.
"""
import pandas as pd


def compute_atr_bands(
    df,
    displace=0,
    factor=1.0,
    length=8,
    price="close",
    average_type="simple",
    true_range_average_type="simple",
):
    """
    Compute ATR Bands: center MA with upper/lower bands at center +/- factor * ATR.

    Requires columns: high, low, close (and price column if not close).
    Adds columns: atr_bands_center, atr_bands_upper, atr_bands_lower.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with OHLC columns
    displace : int
        Bar offset; positive = use value from displace bars ago (default: 0)
    factor : float
        Multiplier for ATR to set band width (default: 1.0)
    length : int
        Period for both center MA and ATR (default: 8)
    price : str
        Price column for center line (default: "close")
    average_type : str
        "simple" or "exponential" - smoothing for center line
    true_range_average_type : str
        "simple", "exponential", or "wilder" - smoothing for ATR

    Returns
    -------
    pd.DataFrame
        DataFrame with atr_bands_center, atr_bands_upper, atr_bands_lower added
    """
    required_cols = ["high", "low", "close"]
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"DataFrame must have columns: {required_cols}")
    if price not in df.columns:
        raise ValueError(f"Price column '{price}' not in DataFrame")

    df = df.copy()
    price_series = df[price]

    # True Range
    high_low = df["high"] - df["low"]
    high_close = abs(df["high"] - df["close"].shift())
    low_close = abs(df["low"] - df["close"].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

    # ATR (moving average of True Range)
    tr_at = true_range_average_type.lower()
    if tr_at == "simple":
        atr = true_range.rolling(window=length).mean()
    elif tr_at == "exponential":
        atr = true_range.ewm(span=length, adjust=False).mean()
    elif tr_at == "wilder":
        atr = true_range.rolling(window=length).mean()
        for i in range(length, len(true_range)):
            atr.iloc[i] = (atr.iloc[i - 1] * (length - 1) + true_range.iloc[i]) / length
    else:
        raise ValueError(f"Unknown true_range_average_type: {true_range_average_type}")

    # Center line (moving average of price)
    shift_val = factor * atr
    avg_type = average_type.lower()
    if avg_type == "simple":
        center = price_series.rolling(window=length).mean()
    elif avg_type == "exponential":
        center = price_series.ewm(span=length, adjust=False).mean()
    else:
        raise ValueError(f"Unknown average_type: {average_type}")

    upper = center + shift_val
    lower = center - shift_val

    # Apply displace (positive = value from displace bars ago)
    if displace != 0:
        center = center.shift(displace)
        upper = upper.shift(displace)
        lower = lower.shift(displace)

    df["atr_bands_center"] = center
    df["atr_bands_upper"] = upper
    df["atr_bands_lower"] = lower
    return df
