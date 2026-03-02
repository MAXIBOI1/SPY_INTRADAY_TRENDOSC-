# data_loader.py
import os
import pandas as pd
import yaml


def load_config(config_path='config.yaml'):
    """Load the YAML config file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML config file '{config_path}': {e}") from e
    except Exception as e:
        raise RuntimeError(f"Error reading config file '{config_path}': {e}") from e
    
    if config is None:
        raise ValueError(f"Config file '{config_path}' is empty or invalid")
    
    return config


def fetch_data(config, date_from=None, date_to=None):
    """
    Load OHLCV data from local parquet file.
    
    Parameters:
    -----------
    config : dict
        Configuration dict with 'data' section
    date_from : str | pd.Timestamp | None, optional
        Start date for filtering (inclusive). Format: "YYYY-MM-DD" or Timestamp.
        If None, no lower bound filtering.
    date_to : str | pd.Timestamp | None, optional
        End date for filtering (exclusive). Format: "YYYY-MM-DD" or Timestamp.
        If None, no upper bound filtering.
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with OHLCV data, optionally filtered by date range
    """
    if not config or 'data' not in config:
        raise ValueError("Config missing 'data' section")
    
    source = config['data'].get('source', 'local_parquet')
    if source != 'local_parquet':
        raise ValueError(f"Only 'local_parquet' source is supported, got '{source}'")

    path = config['data'].get('local_path')
    if not path:
        raise ValueError("Config missing 'data.local_path'")
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")

    try:
        df = pd.read_parquet(path)
    except Exception as e:
        raise RuntimeError(f"Error reading parquet file '{path}': {e}") from e

    if df is None or len(df) == 0:
        raise ValueError(f"Data file '{path}' is empty or could not be loaded")

    required_lower = {'open', 'high', 'low', 'close', 'volume'}
    actual_lower = {col.lower() for col in df.columns}
    missing = required_lower - actual_lower
    if missing:
        raise ValueError(
            f"Missing required columns (case-insensitive): {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    if not isinstance(df.index, pd.DatetimeIndex):
        if 'datetime' in [c.lower() for c in df.columns]:
            dt_col = next(c for c in df.columns if c.lower() == 'datetime')
            try:
                df[dt_col] = pd.to_datetime(df[dt_col])
                df = df.set_index(dt_col)
            except Exception as e:
                raise ValueError(f"Error converting datetime column '{dt_col}': {e}") from e
        else:
            try:
                df.index = pd.to_datetime(df.index)
            except Exception as e:
                raise ValueError(f"Error converting index to datetime: {e}") from e

    df = df.sort_index()
    
    # Apply date filtering if provided
    if date_from is not None or date_to is not None:
        if date_from is not None:
            if isinstance(date_from, str):
                date_from = pd.to_datetime(date_from)
            df = df[df.index >= date_from]
        
        if date_to is not None:
            if isinstance(date_to, str):
                date_to = pd.to_datetime(date_to)
            df = df[df.index < date_to]
    
    return df


def fetch_htf_data(config, timeframe, date_from=None, date_to=None, base_dir=None):
    """
    Load higher-timeframe OHLCV data from pre-built session parquet.

    Path is derived from config['data']['local_path'] by replacing the
    extension with _{timeframe}.parquet (e.g. data/spy_5min_session.parquet
    -> data/spy_5min_session_15min.parquet). If base_dir is provided and
    local_path is not absolute, the path is resolved relative to base_dir.

    Parameters
    ----------
    config : dict
        Configuration dict with 'data' section and 'local_path'
    timeframe : str
        One of "15min", "30min", "1h"
    date_from : str | pd.Timestamp | None, optional
        Start date for filtering (inclusive)
    date_to : str | pd.Timestamp | None, optional
        End date for filtering (exclusive)
    base_dir : str | None, optional
        If provided and local_path is not absolute, resolve path relative to this dir (e.g. strategy root)

    Returns
    -------
    pd.DataFrame
        HTF OHLCV data with datetime index, or None if file not found
    """
    if not config or "data" not in config:
        raise ValueError("Config missing 'data' section")

    allowed = ("15min", "30min", "1h")
    if timeframe not in allowed:
        raise ValueError(f"htf_tmo_timeframe must be one of {allowed}, got {timeframe!r}")

    base_path = config["data"].get("local_path")
    if not base_path:
        raise ValueError("Config missing 'data.local_path'")

    base_path = os.path.normpath(base_path)
    if base_dir is not None and not os.path.isabs(base_path):
        base_path = os.path.join(base_dir, base_path)
    dirname = os.path.dirname(base_path)
    base_name = os.path.splitext(os.path.basename(base_path))[0]
    htf_path = os.path.join(dirname, f"{base_name}_{timeframe}.parquet")

    if not os.path.exists(htf_path):
        return None

    try:
        df = pd.read_parquet(htf_path)
    except Exception as e:
        raise RuntimeError(f"Error reading HTF parquet '{htf_path}': {e}") from e

    if df is None or len(df) == 0:
        return None

    required_lower = {"open", "high", "low", "close", "volume"}
    actual_lower = {col.lower() for col in df.columns}
    missing = required_lower - actual_lower
    if missing:
        raise ValueError(
            f"HTF file '{htf_path}' missing required columns (case-insensitive): {missing}. "
            f"Available: {list(df.columns)}"
        )

    if not isinstance(df.index, pd.DatetimeIndex):
        if "datetime" in [c.lower() for c in df.columns]:
            dt_col = next(c for c in df.columns if c.lower() == "datetime")
            df[dt_col] = pd.to_datetime(df[dt_col])
            df = df.set_index(dt_col)
        else:
            df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    if date_from is not None or date_to is not None:
        if date_from is not None:
            if isinstance(date_from, str):
                date_from = pd.to_datetime(date_from)
            df = df[df.index >= date_from]
        if date_to is not None:
            if isinstance(date_to, str):
                date_to = pd.to_datetime(date_to)
            df = df[df.index < date_to]

    return df
