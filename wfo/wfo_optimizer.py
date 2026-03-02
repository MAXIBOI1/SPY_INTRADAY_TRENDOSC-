# wfo_optimizer.py
"""
Optuna-based optimization for walk-forward optimization.
Optimizes parameters for a single training period.
"""
import copy
import logging
import math
import tempfile

import numpy as np
import optuna
import pandas as pd

from backtester import run_backtest

log = logging.getLogger("wfo_optimizer")

# Default parameter ranges (used if not specified in config). Blank slate: empty.
DEFAULT_RANGES = {}

# Default categorical parameter options (used if not specified in config). Blank slate: empty.
DEFAULT_CATEGORICAL = {}

# Parameter types: "int" or "float"
PARAMETER_TYPES = {
    "hilopro_long_threshold": "int",
    "hilopro_short_threshold": "int",
    "atr_bands_factor": "float",
    "atr_bands_length": "int",
    "max_trades_per_session": "int",
    "stop_adjustment_factor": "float",
    "atr_multiplier_breakeven": "float",
}

# Parameter step sizes for float parameters (None = continuous, or step size for discrete)
# Used by Optuna's suggest_float step parameter
PARAMETER_STEPS = {
    "atr_bands_factor": 0.1,
    "stop_adjustment_factor": None,
    "atr_multiplier_breakeven": None,
    "atr_multiplier_target": 0.1,
}

# Default composite weights when wfo.optimization.composite_weights is not set. V01-aligned (sum = 1.0).
DEFAULT_COMPOSITE_WEIGHTS = {
    "expectancy": 0.25,
    "max_drawdown_pct": 0.18,
    "sharpe_ratio": 0.20,
    "win_rate_pct": 0.28,
    "trades_per_month": 0.04,
    "breakeven_exits_pct": 0.05,
}


def _chunk_returns_from_curve(portfolio_curve, train_start, train_end, n_chunks):
    """
    Compute per-chunk returns from equity curve over the training window.

    Parameters:
    -----------
    portfolio_curve : pd.Series
        Equity curve with datetime index (from metrics["_portfolio_curve"]).
    train_start : pd.Timestamp
        Start of training window (inclusive).
    train_end : pd.Timestamp
        End of training window (exclusive).
    n_chunks : int
        Number of equal time spans to split the window into.

    Returns:
    --------
    list of float
        Chunk returns as decimals (e.g. 0.05 = 5%). May be shorter than n_chunks if some chunks have no data.
    """
    if portfolio_curve is None or portfolio_curve.empty or n_chunks < 1:
        return []
    curve = portfolio_curve[(portfolio_curve.index >= train_start) & (portfolio_curve.index < train_end)]
    if curve.empty or len(curve) < 2:
        return []
    curve = curve.sort_index()
    start_ts = curve.index.min()
    end_ts = curve.index.max()
    if start_ts >= end_ts:
        return []
    chunk_returns = []
    for i in range(n_chunks):
        t0 = start_ts + (end_ts - start_ts) * i / n_chunks
        t1 = start_ts + (end_ts - start_ts) * (i + 1) / n_chunks
        in_chunk = curve[(curve.index >= t0) & (curve.index < t1)]
        if in_chunk.empty:
            continue
        start_val = float(curve[curve.index >= t0].iloc[0]) if not curve[curve.index >= t0].empty else None
        end_val = float(curve[curve.index <= in_chunk.index.max()].iloc[-1]) if not curve[curve.index <= in_chunk.index.max()].empty else None
        if start_val is None or end_val is None or start_val <= 0:
            continue
        chunk_returns.append((end_val - start_val) / start_val)
    return chunk_returns


def _consistency_score(chunk_returns, mode):
    """
    Convert chunk returns into a 0-100 consistency term for blending with composite score.

    Parameters:
    -----------
    chunk_returns : list of float
        Per-chunk returns as decimals (from _chunk_returns_from_curve).
    mode : str
        "maximin" = score from minimum chunk return; "variance_penalty" = reward low variance.

    Returns:
    --------
    float
        Score in [0, 100] for blending. Higher = more consistent / less bad worst chunk.
    """
    if not chunk_returns:
        return 0.0
    if mode == "maximin":
        min_ret = min(chunk_returns)
        # Map min return (decimal) to 0-100: -100% -> 0, 0% -> 50, +100% -> 100
        return float(np.clip((min_ret * 100 + 100) / 2, 0, 100))
    if mode == "variance_penalty":
        std_ret = float(np.std(chunk_returns))
        # Reward low variance: 100 - k * std (in %). k=5 => std 10% -> score 50
        return float(np.clip(100 - 5 * (std_ret * 100), 0, 100))
    return 0.0


def _param_stability_penalty(trial_params, previous_best_params, merged_ranges):
    """
    Normalized distance between trial params and previous best (0 = identical, higher = more different).
    Numeric params: L2 on normalized [0,1] range. Categorical: 0 if same else 1.
    """
    if not previous_best_params:
        return 0.0
    total = 0.0
    count = 0
    for param_name in trial_params:
        if param_name not in previous_best_params:
            continue
        trial_val = trial_params[param_name]
        prev_val = previous_best_params[param_name]
        param_range = merged_ranges.get(param_name) if merged_ranges else None
        if isinstance(param_range, list) and len(param_range) == 2 and not (isinstance(trial_val, str) or isinstance(prev_val, str)):
            try:
                lo, hi = float(param_range[0]), float(param_range[1])
                if hi != lo:
                    t_norm = (float(trial_val) - lo) / (hi - lo)
                    p_norm = (float(prev_val) - lo) / (hi - lo)
                    total += (t_norm - p_norm) ** 2
            except (TypeError, ValueError):
                total += 0.0 if trial_val == prev_val else 1.0
        else:
            total += 0.0 if trial_val == prev_val else 1.0
        count += 1
    if count == 0:
        return 0.0
    return math.sqrt(total / count)


def _get_composite_scorer_registry():
    """Build registry of metric key -> scorer function (V01: 6 metrics from scorers)."""
    from scorers import (
        score_breakeven_exits_pct,
        score_expectancy,
        score_max_drawdown_pct,
        score_sharpe_ratio,
        score_trades_per_month,
        score_win_rate_pct,
    )
    return {
        "expectancy": score_expectancy,
        "max_drawdown_pct": score_max_drawdown_pct,
        "sharpe_ratio": score_sharpe_ratio,
        "win_rate_pct": score_win_rate_pct,
        "trades_per_month": score_trades_per_month,
        "breakeven_exits_pct": score_breakeven_exits_pct,
    }


def calculate_simplified_composite_score(metrics, base_config):
    """
    Calculate composite score for optimization from configurable weighted metrics.
    Uses wfo.optimization.composite_weights if set; otherwise DEFAULT_COMPOSITE_WEIGHTS.
    
    Parameters:
    -----------
    metrics : dict
        Metrics dict from backtest (may include train_period_months, min/max_trades_per_month for trades_per_month)
    base_config : dict
        Base configuration dict (for cfg_timeframe, initial_capital, and optional composite_weights)
    
    Returns:
    --------
    float
        Composite score (0-100), compatible with ranking system
    """
    row = dict(metrics)
    if "cfg_timeframe" not in row:
        row["cfg_timeframe"] = base_config.get("data", {}).get("timeframe", "15m")
    if "initial_capital" not in row:
        row["initial_capital"] = base_config.get("portfolio", {}).get("initial_capital", 10000)
    
    composite_weights = base_config.get("wfo", {}).get("optimization", {}).get("composite_weights")
    if not composite_weights:
        composite_weights = DEFAULT_COMPOSITE_WEIGHTS
    
    registry = _get_composite_scorer_registry()
    weighted_sum = 0.0
    total_weight = 0.0
    
    for key, weight in composite_weights.items():
        weight = float(weight) if weight is not None else 0.0
        if weight <= 0:
            continue
        fn = registry.get(key)
        if fn is None:
            continue
        try:
            score, _ = fn(row)
            weighted_sum += score * weight
            total_weight += weight
        except Exception:
            continue
    
    return weighted_sum / total_weight if total_weight > 0 else 0.0


def optimize_period(
    base_config,
    train_df,
    objective_metric="avg_trade_return_pct",
    n_trials=100,
    min_trades=10,
    max_trades=None,
    parameter_ranges=None,
    max_drawdown_pct_cap=None,
    min_win_rate_pct=None,
    train_start=None,
    train_end=None,
    consistency_subperiods=0,
    consistency_weight=0.0,
    consistency_mode="maximin",
    previous_best_params=None,
    param_stability_penalty_weight=0.0,
    **optuna_kwargs,
):
    """
    Optimize parameters for a single training period using Optuna.

    Parameters:
    -----------
    base_config : dict
        Base configuration dict (will be modified with trial parameters)
    train_df : pd.DataFrame
        Training DataFrame with indicators already calculated
    objective_metric : str, optional
        Metric to optimize (default: "avg_trade_return_pct")
    n_trials : int, optional
        Number of Optuna trials (default: 100)
    min_trades : int, optional
        Minimum trades required for valid optimization (default: 10)
    max_trades : int | None, optional
        Maximum trades allowed (default: None, no limit). Strategies exceeding this are penalized.
    parameter_ranges : dict | None, optional
        Dict mapping parameter names to [min, max] ranges. If None, uses DEFAULT_RANGES.
    max_drawdown_pct_cap : float | None, optional
        Reject trials where training-period max_drawdown_pct is worse than this (e.g. -40 for 40% cap). None = no cap.
    min_win_rate_pct : float | None, optional
        Reject trials where training-period win_rate_pct is below this (e.g. 45 for 45% minimum). None = no minimum.
    train_start : pd.Timestamp | None, optional
        Start of training window (for consistency chunking). Required when consistency_weight > 0.
    train_end : pd.Timestamp | None, optional
        End of training window (for consistency chunking).
    consistency_subperiods : int, optional
        Number of chunks to split train window into for consistency scoring; 0 = disabled.
    consistency_weight : float, optional
        Blend (1-w)*composite + w*consistency_term; 0 = disabled.
    consistency_mode : str, optional
        "maximin" or "variance_penalty" for consistency term.
    previous_best_params : dict | None, optional
        Best params from previous WFO period (for stability penalty).
    param_stability_penalty_weight : float, optional
        Penalty weight for param distance from previous best; 0 = disabled.
    **optuna_kwargs
        Additional arguments to pass to optuna.create_study()

    Returns:
    --------
    tuple
        (best_params: dict, best_value: float, study: optuna.Study)
    """
    # Merge parameter ranges with defaults
    if parameter_ranges is None:
        parameter_ranges = {}
    
    # Separate numeric ranges from categorical options
    # Categorical parameters are lists of strings (not [min, max] pairs)
    categorical_params = {}
    numeric_ranges = {}
    
    for param_name, param_value in parameter_ranges.items():
        # Check if it's a categorical (list of strings) or numeric range ([min, max])
        if isinstance(param_value, list) and len(param_value) > 0:
            # If first element is a string, treat as categorical
            if isinstance(param_value[0], str):
                categorical_params[param_name] = param_value
            else:
                # Numeric range [min, max]
                numeric_ranges[param_name] = param_value
        else:
            log.warning(f"Invalid parameter format for {param_name}: {param_value}. Skipping.")
    
    # Use defaults for parameters not specified in config
    merged_ranges = DEFAULT_RANGES.copy()
    merged_ranges.update(numeric_ranges)
    
    # Merge categorical parameters with defaults
    merged_categorical = DEFAULT_CATEGORICAL.copy()
    merged_categorical.update(categorical_params)
    
    def objective(trial):
        # Create a copy of base config to modify
        trial_config = copy.deepcopy(base_config)
        params = trial_config["strategy"]["params"]
        
        # Define search space using configurable ranges
        # Skip atr_multiplier_target here - it's handled conditionally later based on exit_target_strategy
        for param_name, param_range in merged_ranges.items():
            # Skip conditional parameters that are handled separately
            if param_name == "atr_multiplier_target":
                continue
            
            if not isinstance(param_range, list) or len(param_range) != 2:
                log.warning(f"Invalid range for {param_name}: {param_range}. Using default.")
                if param_name in DEFAULT_RANGES:
                    param_range = DEFAULT_RANGES[param_name]
                else:
                    continue
            
            min_val, max_val = param_range[0], param_range[1]
            
            if min_val >= max_val:
                log.warning(f"Invalid range for {param_name}: min >= max. Using default.")
                if param_name in DEFAULT_RANGES:
                    min_val, max_val = DEFAULT_RANGES[param_name][0], DEFAULT_RANGES[param_name][1]
                else:
                    continue
            
            # Determine parameter type
            param_type = PARAMETER_TYPES.get(param_name, "int")
            
            if param_type == "float":
                # Get step size from PARAMETER_STEPS (None = continuous)
                step = PARAMETER_STEPS.get(param_name, None)
                params[param_name] = trial.suggest_float(param_name, min_val, max_val, step=step)
            else:
                # Integer parameters
                if not isinstance(min_val, int) or not isinstance(max_val, int):
                    log.warning(f"Non-integer range for integer parameter {param_name}. Converting to int.")
                    min_val, max_val = int(min_val), int(max_val)
                params[param_name] = trial.suggest_int(param_name, min_val, max_val)
        
        # Categorical parameters (read from config)
        for param_name, options in merged_categorical.items():
            if not isinstance(options, list) or len(options) == 0:
                log.warning(f"Invalid categorical options for {param_name}: {options}. Using default.")
                options = DEFAULT_CATEGORICAL.get(param_name, [])
            if len(options) > 0:
                params[param_name] = trial.suggest_categorical(param_name, options)
            else:
                log.warning(f"No options available for categorical parameter {param_name}. Skipping.")
        
        # ATR target multiplier (only when exit_target_strategy is "atr" or "hybrid" and we are optimizing it)
        if params.get("exit_target_strategy") in ["atr", "hybrid"] and "atr_multiplier_target" in merged_ranges:
            atr_target_range = merged_ranges.get("atr_multiplier_target", [1.5, 4.0])
            if isinstance(atr_target_range, list) and len(atr_target_range) == 2:
                min_target, max_target = float(atr_target_range[0]), float(atr_target_range[1])
            else:
                min_target, max_target = 1.5, 4.0
            # Get step size from PARAMETER_STEPS
            step = PARAMETER_STEPS.get("atr_multiplier_target", None)
            params["atr_multiplier_target"] = trial.suggest_float(
                "atr_multiplier_target", min_target, max_target, step=step
            )
        
        # Run backtest on training data
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                metrics = run_backtest(
                    config=trial_config,
                    df=train_df.copy(),
                    skip_indicators=False,  # Recalculate indicators with trial parameters
                    output_dir=temp_dir,
                    output_suffix=f"trial_{trial.number}",
                )
            
            if metrics is None:
                # No data/trades - return large positive penalty (Optuna minimizes)
                return 1e6
            
            num_trades = metrics.get("num_trades", 0)
            if num_trades < min_trades:
                # Insufficient trades - return positive penalty proportional to shortage
                penalty = 1e6 * (1 + (min_trades - num_trades) / min_trades)
                return penalty
            
            if max_trades is not None and num_trades > max_trades:
                # Too many trades - penalize as strongly as undertrading (enforce selectivity)
                penalty = 1e6 * (1 + (num_trades - max_trades) / max_trades)
                return penalty
            
            if max_drawdown_pct_cap is not None and metrics.get("max_drawdown_pct", 0) < max_drawdown_pct_cap:
                return 1e6
            
            if min_win_rate_pct is not None:
                win_rate = metrics.get("win_rate_pct", 0)
                if win_rate < min_win_rate_pct:
                    return 1e6
            
            # Check if objective_metric is "composite_score" or use single metric
            if objective_metric == "composite_score":
                # Enrich metrics for composite: train period length and trades-per-month band (for score_trades_per_month)
                if not train_df.empty and train_df.index is not None and len(train_df.index) > 0:
                    train_period_months = (train_df.index[-1] - train_df.index[0]).days / 30.44
                    metrics["train_period_months"] = train_period_months
                opt_cfg = base_config.get("wfo", {}).get("optimization", {})
                if opt_cfg.get("min_trades_per_month") is not None:
                    metrics["min_trades_per_month"] = opt_cfg["min_trades_per_month"]
                if opt_cfg.get("max_trades_per_month") is not None:
                    metrics["max_trades_per_month"] = opt_cfg["max_trades_per_month"]
                # Use config-driven composite score for optimization
                composite_score = calculate_simplified_composite_score(metrics, base_config)
                if composite_score <= 0:
                    return 1e6
                raw_score = composite_score  # may be blended with consistency term below
                # Consistency-aware blend: prefer params that perform steadily across sub-periods
                if (
                    consistency_weight > 0
                    and consistency_subperiods >= 1
                    and train_start is not None
                    and train_end is not None
                ):
                    curve = metrics.get("_portfolio_curve")
                    if curve is not None and not (hasattr(curve, "empty") and curve.empty):
                        ts_start = pd.Timestamp(train_start) if not isinstance(train_start, pd.Timestamp) else train_start
                        ts_end = pd.Timestamp(train_end) if not isinstance(train_end, pd.Timestamp) else train_end
                        chunk_returns = _chunk_returns_from_curve(curve, ts_start, ts_end, consistency_subperiods)
                        if chunk_returns:
                            consistency_term = _consistency_score(chunk_returns, consistency_mode)
                            raw_score = (1 - consistency_weight) * composite_score + consistency_weight * consistency_term
            else:
                # Original single-metric optimization
                objective_value = metrics.get(objective_metric)
                if objective_value is None:
                    log.warning(f"Objective metric '{objective_metric}' not found in metrics")
                    return 1e6
                
                # Handle NaN, inf, or invalid values
                try:
                    objective_float = float(objective_value)
                    if not math.isfinite(objective_float):
                        log.warning(f"Objective metric '{objective_metric}' is not finite: {objective_value}")
                        return 1e6
                except (ValueError, TypeError):
                    log.warning(f"Objective metric '{objective_metric}' cannot be converted to float: {objective_value}")
                    return 1e6
                
                raw_score = objective_float

            # Parameter stability penalty: penalize large changes from previous period
            stability_penalty = 0.0
            if param_stability_penalty_weight > 0 and previous_best_params is not None:
                stability_penalty = _param_stability_penalty(params, previous_best_params, merged_ranges)
            # Optuna minimizes: return -score + penalty so higher score wins, penalty hurts
            return -raw_score + param_stability_penalty_weight * stability_penalty
            
        except Exception as e:
            log.error(f"Error in trial {trial.number}: {e}")
            return 1e6
    
    # Create Optuna study
    study = optuna.create_study(direction="minimize", **optuna_kwargs)
    
    log.info(f"Starting optimization with {n_trials} trials...")
    study.optimize(objective, n_trials=n_trials)
    
    # Extract best parameters
    best_params = study.best_params.copy()
    best_value = -study.best_value  # Negate back to get actual value
    
    log.info(f"Optimization complete. Best {objective_metric}: {best_value:.4f}")
    log.info(f"Best parameters: {best_params}")
    
    return best_params, best_value, study


def validate_params(config_with_params, test_df, date_from=None, date_to=None):
    """
    Validate optimized parameters on test period.
    
    Parameters:
    -----------
    config_with_params : dict
        Configuration dict with optimized parameters
    test_df : pd.DataFrame
        Test DataFrame with indicators already calculated
    date_from : str | pd.Timestamp | None, optional
        Start date for filtering test_df
    date_to : str | pd.Timestamp | None, optional
        End date for filtering test_df
    
    Returns:
    --------
    dict
        Metrics dict from backtest, includes _trades_df for equity curve
    """
    # Slice test_df to window if dates provided
    test_window_df = test_df.copy()
    if date_from is not None:
        if isinstance(date_from, str):
            date_from = pd.to_datetime(date_from)
        test_window_df = test_window_df[test_window_df.index >= date_from]
    
    if date_to is not None:
        if isinstance(date_to, str):
            date_to = pd.to_datetime(date_to)
        test_window_df = test_window_df[test_window_df.index < date_to]
    
    if len(test_window_df) == 0:
        log.warning("Test window is empty after filtering")
        return None
    
    # Run backtest on test period
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            metrics = run_backtest(
                config=config_with_params,
                df=test_window_df,
                skip_indicators=False,  # Recalculate indicators with optimized parameters
                output_dir=temp_dir,
                output_suffix="validation",
            )
        return metrics
    except Exception as e:
        log.error(f"Error validating parameters: {e}")
        return None
