# indicators/ema.py
import pandas as pd

def compute_ema(df, period, column_prefix='ema_'):
    """Compute a single EMA on 'close'. Adds column ema_<period>."""
    if 'close' not in df.columns:
        raise ValueError("DataFrame must have 'close' column")
    column_name = f"{column_prefix}{period}"
    df[column_name] = df['close'].ewm(span=period, adjust=False).mean()
    return df
