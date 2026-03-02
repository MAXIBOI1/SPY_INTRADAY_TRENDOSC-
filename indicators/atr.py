# indicators/atr.py
import pandas as pd

def compute_atr(df, period=14, method='wilder'):
    """Compute ATR. Requires high, low, close. Adds column 'atr'."""
    required_cols = ['high', 'low', 'close']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"DataFrame must have columns: {required_cols}")
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift())
    low_close = abs(df['low'] - df['close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    if method.lower() == 'simple':
        atr = true_range.rolling(window=period).mean()
    elif method.lower() == 'exponential':
        atr = true_range.ewm(span=period, adjust=False).mean()
    elif method.lower() == 'wilder':
        atr = true_range.rolling(window=period).mean()
        for i in range(period, len(true_range)):
            atr.iloc[i] = (atr.iloc[i-1] * (period - 1) + true_range.iloc[i]) / period
    else:
        raise ValueError(f"Unknown ATR method: {method}")
    df['atr'] = atr
    return df
