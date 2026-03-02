# wfo_framework.py
"""
Walk-forward optimization framework.
Manages rolling windows and orchestrates optimization + validation.
"""
import logging
import math

import numpy as np
import pandas as pd

from .wfo_optimizer import optimize_period, validate_params

log = logging.getLogger("wfo_framework")


def generate_rolling_windows(data_start, data_end, train_months, test_months, step_months):
    """
    Generate rolling window tuples for walk-forward optimization.
    
    Parameters:
    -----------
    data_start : pd.Timestamp
        Start of available data
    data_end : pd.Timestamp
        End of available data
    train_months : int
        Training period length in months
    test_months : int
        Test period length in months
    step_months : int
        Step size in months (how much to roll forward each window)
    
    Returns:
    --------
    list
        List of (train_start, train_end, test_start, test_end) tuples
    """
    windows = []
    current_start = data_start
    
    while current_start < data_end:
        train_end = current_start + pd.DateOffset(months=train_months)
        test_start = train_end
        test_end = test_start + pd.DateOffset(months=test_months)
        
        # Only add window if we have enough data
        if test_end <= data_end:
            windows.append((current_start, train_end, test_start, test_end))
            log.debug(f"Generated window {len(windows)}: Train {current_start.date()} to {train_end.date()}, Test {test_start.date()} to {test_end.date()}")
        
        # Roll forward
        current_start = current_start + pd.DateOffset(months=step_months)
        
        # Termination condition: if we can't create a valid window, break naturally
        # The while condition (current_start < data_end) and test_end check above handle this
    
    return windows


def build_combined_oos_equity_curve(results_list, initial_capital):
    """
    Build combined out-of-sample equity curve from all test periods.
    Uses _portfolio_curve for continuous bar-level data (not just trade exits).

    Parameters:
    -----------
    results_list : list
        List of result dicts, each containing test_metrics with _portfolio_curve
    initial_capital : float
        Initial capital for portfolio

    Returns:
    --------
    pd.DataFrame
        DataFrame with columns: date, portfolio_value, cumulative_return_pct
    """
    curve_parts = []
    running_capital = initial_capital

    for result in results_list:
        test_metrics = result.get("test_metrics")
        if test_metrics is None:
            continue

        portfolio_curve = test_metrics.get("_portfolio_curve")
        if portfolio_curve is None or portfolio_curve.empty:
            continue

        test_start = result.get("test_start")
        test_end = result.get("test_end")
        if test_start is not None:
            test_start = pd.Timestamp(test_start)
        if test_end is not None:
            test_end = pd.Timestamp(test_end)

        # Slice to test window only
        if test_start is not None:
            portfolio_curve = portfolio_curve[portfolio_curve.index >= test_start]
        if test_end is not None:
            portfolio_curve = portfolio_curve[portfolio_curve.index < test_end]
        if portfolio_curve.empty:
            continue

        # Additive gains (no compound): add this period's dollar gain (vs initial_capital) to running total
        curve_scaled = running_capital + (portfolio_curve - initial_capital)
        curve_parts.append(curve_scaled)
        running_capital = float(curve_scaled.iloc[-1])

    if not curve_parts:
        return pd.DataFrame(columns=["date", "portfolio_value", "cumulative_return_pct"])

    combined = pd.concat(curve_parts)
    combined = combined[~combined.index.duplicated(keep="first")]
    combined = combined.sort_index()

    equity_curve_df = pd.DataFrame(
        {"portfolio_value": combined},
        index=combined.index,
    )
    equity_curve_df["cumulative_return_pct"] = (
        (equity_curve_df["portfolio_value"] - initial_capital) / initial_capital
    ) * 100
    equity_curve_df.index.name = "date"
    return equity_curve_df


def aggregate_wfo_results(results_list):
    """
    Aggregate metrics across all periods.
    
    Parameters:
    -----------
    results_list : list
        List of result dicts from each period
    
    Returns:
    --------
    dict
        Aggregated summary dict
    """
    if not results_list:
        return {}
    
    # Extract OOS metrics
    oos_returns = []
    oos_sharpes = []
    oos_calmars = []
    oos_max_dds = []
    oos_num_trades = []
    oos_profit_factors = []
    periods_with_insufficient_trades = 0
    
    for result in results_list:
        test_metrics = result.get("test_metrics")
        if test_metrics is None:
            continue
        
        # Count periods with insufficient test trades
        if result.get("test_trades_insufficient", False):
            periods_with_insufficient_trades += 1
        
        oos_returns.append(test_metrics.get("return_pct", 0.0))
        oos_sharpes.append(test_metrics.get("sharpe_ratio"))
        oos_calmars.append(test_metrics.get("calmar_ratio"))
        oos_max_dds.append(test_metrics.get("max_drawdown_pct", 0.0))
        oos_num_trades.append(test_metrics.get("num_trades", 0))
        oos_profit_factors.append(test_metrics.get("profit_factor"))
    
    # Filter out None values for metrics that might be None
    oos_sharpes_clean = [x for x in oos_sharpes if x is not None]
    oos_calmars_clean = [x for x in oos_calmars if x is not None]
    oos_profit_factors_clean = [x for x in oos_profit_factors if x is not None and math.isfinite(x)]
    
    # Calculate parameter stability
    param_stability = calculate_param_stability(results_list)
    
    # OOS return by period (for consistency reporting)
    oos_return_by_period = []
    for result in results_list:
        tm = result.get("test_metrics")
        if tm is None:
            continue
        ts_start = result.get("test_start")
        ts_end = result.get("test_end")
        oos_return_by_period.append({
            "period_idx": result.get("period_idx"),
            "test_start": ts_start.isoformat() if hasattr(ts_start, "isoformat") else str(ts_start),
            "test_end": ts_end.isoformat() if hasattr(ts_end, "isoformat") else str(ts_end),
            "return_pct": float(tm.get("return_pct", 0.0)),
        })
    
    # OOS return by year: attribute each period to year of test_start, then average return per year
    year_to_returns = {}
    for result in results_list:
        tm = result.get("test_metrics")
        if tm is None:
            continue
        ts_start = result.get("test_start")
        if hasattr(ts_start, "year"):
            year = ts_start.year
        else:
            year = pd.Timestamp(ts_start).year
        year_to_returns.setdefault(year, []).append(float(tm.get("return_pct", 0.0)))
    oos_return_by_year = [
        {"year": year, "return_pct": float(np.mean(year_to_returns[year]))}
        for year in sorted(year_to_returns)
    ]
    
    summary = {
        "num_periods": len(results_list),
        "periods_with_insufficient_test_trades": periods_with_insufficient_trades,
        "oos_return_by_period": oos_return_by_period,
        "oos_return_by_year": oos_return_by_year,
        "oos_return_mean": float(np.mean(oos_returns)) if oos_returns else 0.0,
        "oos_return_std": float(np.std(oos_returns)) if oos_returns else 0.0,
        "oos_return_min": float(np.min(oos_returns)) if oos_returns else 0.0,
        "oos_return_max": float(np.max(oos_returns)) if oos_returns else 0.0,
        "oos_return_median": float(np.median(oos_returns)) if oos_returns else 0.0,
        "oos_sharpe_mean": float(np.mean(oos_sharpes_clean)) if oos_sharpes_clean else None,
        "oos_sharpe_std": float(np.std(oos_sharpes_clean)) if oos_sharpes_clean else None,
        "oos_calmar_mean": float(np.mean(oos_calmars_clean)) if oos_calmars_clean else None,
        "oos_calmar_std": float(np.std(oos_calmars_clean)) if oos_calmars_clean else None,
        "oos_max_dd_mean": float(np.mean(oos_max_dds)) if oos_max_dds else 0.0,
        "oos_max_dd_std": float(np.std(oos_max_dds)) if oos_max_dds else 0.0,
        "oos_total_trades": int(sum(oos_num_trades)) if oos_num_trades else 0,
        "oos_profit_factor_mean": float(np.mean(oos_profit_factors_clean)) if oos_profit_factors_clean else None,
        "param_stability_score": param_stability,
    }
    
    return summary


def calculate_param_stability(results_list):
    """
    Calculate parameter stability score across periods.
    
    Parameters:
    -----------
    results_list : list
        List of result dicts with best_params
    
    Returns:
    --------
    float
        Stability score (0-1, higher = more stable)
    """
    if len(results_list) < 2:
        return 1.0
    
    # Extract parameter values
    param_values = {}
    for result in results_list:
        best_params = result.get("best_params", {})
        for key, value in best_params.items():
            if key not in param_values:
                param_values[key] = []
            param_values[key].append(value)
    
    if not param_values:
        return 1.0
    
    # Calculate coefficient of variation for each parameter
    cvs = []
    for key, values in param_values.items():
        if len(values) < 2:
            continue
        
        # For categorical parameters, calculate percentage agreement
        if isinstance(values[0], str):
            most_common = max(set(values), key=values.count)
            agreement = values.count(most_common) / len(values)
            stability = agreement  # 1.0 = all same, 0.0 = all different
        else:
            # For numeric parameters, use CV
            mean_val = np.mean(values)
            std_val = np.std(values)
            if mean_val != 0:
                cv = std_val / abs(mean_val)
                stability = 1.0 / (1.0 + cv)  # Convert CV to stability (0-1)
            else:
                stability = 1.0 if std_val == 0 else 0.0
        
        cvs.append(stability)
    
    return float(np.mean(cvs)) if cvs else 1.0


def calculate_wfo_consistency(ranked_periods, top_n=5):
    """
    Calculate walk-forward consistency metrics.
    Tracks how often top-ranked periods appear consecutively.
    
    Parameters:
    -----------
    ranked_periods : list
        List of ranked period dicts (from rank_wfo_periods)
    top_n : int, optional
        Top N periods to consider for consistency (default: 5)
    
    Returns:
    --------
    dict
        Consistency metrics:
        - top_n_consecutive_count: Number of times top N periods appear consecutively
        - top_n_consecutive_ratio: Ratio of consecutive appearances to total possible
        - rank_stability: Average rank change between consecutive periods
        - top_period_consistency: How often period 0 (top-ranked) appears in top N
    """
    if len(ranked_periods) < 2:
        return {
            "top_n_consecutive_count": 0,
            "top_n_consecutive_ratio": 0.0,
            "rank_stability": 0.0,
            "top_period_consistency": 0.0,
        }
    
    # Create period_idx -> rank mapping
    period_to_rank = {p["period_idx"]: p["rank"] for p in ranked_periods}
    
    # Get top N period indices
    top_n_periods = set(p["period_idx"] for p in ranked_periods[:top_n])
    
    # Calculate consecutive appearances of top N periods
    consecutive_count = 0
    total_possible = len(ranked_periods) - 1  # Number of consecutive pairs
    
    # Sort periods by period_idx to check consecutive periods
    sorted_periods = sorted(ranked_periods, key=lambda x: x["period_idx"])
    
    for i in range(len(sorted_periods) - 1):
        current_idx = sorted_periods[i]["period_idx"]
        next_idx = sorted_periods[i + 1]["period_idx"]
        
        # Check if both are in top N
        if current_idx in top_n_periods and next_idx in top_n_periods:
            consecutive_count += 1
    
    consecutive_ratio = consecutive_count / total_possible if total_possible > 0 else 0.0
    
    # Calculate rank stability (average rank change between consecutive periods)
    rank_changes = []
    for i in range(len(sorted_periods) - 1):
        current_rank = sorted_periods[i]["rank"]
        next_rank = sorted_periods[i + 1]["rank"]
        rank_change = abs(current_rank - next_rank)
        rank_changes.append(rank_change)
    
    avg_rank_change = sum(rank_changes) / len(rank_changes) if rank_changes else 0.0
    
    # Calculate how often top-ranked period appears in top N
    top_period_idx = ranked_periods[0]["period_idx"]
    top_period_in_top_n = 1 if top_period_idx in top_n_periods else 0
    top_period_consistency = top_period_in_top_n  # This is always 1.0 since top period is always in top N
    
    # Better metric: how stable is the top-ranked period across all periods?
    # Count how many times the top-ranked period appears in top N when considering
    # rolling windows or how consistent top performers are
    top_period_consistency = 1.0  # Top period is always in top N by definition
    
    return {
        "top_n_consecutive_count": consecutive_count,
        "top_n_consecutive_ratio": consecutive_ratio,
        "rank_stability": avg_rank_change,
        "top_period_consistency": top_period_consistency,
        "top_n": top_n,
    }


def find_best_params_per_metric(results_list):
    """
    Find the period that had the best OOS value for each metric.

    Parameters:
    -----------
    results_list : list
        List of result dicts with test_metrics and best_params

    Returns:
    --------
    dict
        Keys: metric names. Values: {"period_idx": int, "value": float, "params": dict}
    """
    if not results_list:
        return {}

    # (metric_key, higher_is_better, label for output)
    metrics_config = [
        ("return_pct", True, "return_pct"),
        ("sharpe_ratio", True, "sharpe_ratio"),
        ("calmar_ratio", True, "calmar_ratio"),
        ("profit_factor", True, "profit_factor"),
        ("max_drawdown_pct", True, "max_drawdown_pct"),  # least negative is best (max)
    ]

    out = {}
    for metric_key, higher_is_better, label in metrics_config:
        best_idx = None
        best_value = None
        best_params = None

        for result in results_list:
            test_metrics = result.get("test_metrics")
            if test_metrics is None:
                continue
            value = test_metrics.get(metric_key)
            if value is None:
                continue
            if math.isfinite(value) is False:
                continue
            if best_value is None:
                best_idx = result["period_idx"]
                best_value = value
                best_params = result.get("best_params", {})
                continue
            if higher_is_better and value > best_value:
                best_idx = result["period_idx"]
                best_value = value
                best_params = result.get("best_params", {})
            elif not higher_is_better and value < best_value:
                best_idx = result["period_idx"]
                best_value = value
                best_params = result.get("best_params", {})

        if best_idx is not None:
            out[label] = {
                "period_idx": best_idx,
                "value": float(best_value),
                "params": best_params,
            }
    return out


def find_best_period_by_objective(results_list, objective_metric):
    """
    Find the period with the highest objective metric score (best_value from optimization).

    Parameters:
    -----------
    results_list : list
        List of result dicts with best_value and test_metrics
    objective_metric : str
        Name of the objective metric (e.g. "sharpe_ratio")

    Returns:
    --------
    dict
        {"period_idx": int, "best_value": float, "test_metrics": dict, "params": dict,
         "test_start", "test_end"} or empty dict if no results
    """
    if not results_list:
        return {}

    best_result = max(results_list, key=lambda r: r.get("best_value") or float("-inf"))
    ts_start = best_result.get("test_start")
    ts_end = best_result.get("test_end")
    return {
        "period_idx": best_result["period_idx"],
        "best_value": float(best_result["best_value"]),
        "test_metrics": best_result.get("test_metrics") or {},
        "params": best_result.get("best_params") or {},
        "test_start": ts_start.isoformat() if hasattr(ts_start, "isoformat") else str(ts_start),
        "test_end": ts_end.isoformat() if hasattr(ts_end, "isoformat") else str(ts_end),
    }


def run_wfo(base_config, wfo_config, data_df_with_indicators):
    """
    Run walk-forward optimization.
    
    Parameters:
    -----------
    base_config : dict
        Base YAML config dict
    wfo_config : dict
        WFO settings dict with train_months, test_months, step_months, optimization settings
    data_df_with_indicators : pd.DataFrame
        Full DataFrame with all indicators already calculated
    
    Returns:
    --------
    dict
        Results dict with per-period results, summary, and combined equity curve
    """
    if data_df_with_indicators.empty:
        raise ValueError("DataFrame is empty")
    
    # Extract WFO settings
    train_months = wfo_config.get("train_months", 3)
    test_months = wfo_config.get("test_months", 1)
    step_months = wfo_config.get("step_months", 1)
    opt_config = wfo_config.get("optimization", {})
    n_trials = opt_config.get("n_trials", 100)
    objective_metric = opt_config.get("objective", "avg_trade_return_pct")
    min_trades = opt_config.get("min_trades", 10)
    max_trades = opt_config.get("max_trades", None)
    if opt_config.get("min_trades_per_month") is not None:
        min_trades = int(opt_config["min_trades_per_month"] * train_months)
        log.info(f"Using min_trades_per_month: derived min_trades={min_trades} (train_months={train_months})")
    if opt_config.get("max_trades_per_month") is not None:
        max_trades = int(opt_config["max_trades_per_month"] * train_months)
        log.info(f"Using max_trades_per_month: derived max_trades={max_trades} (train_months={train_months})")
    max_drawdown_pct_cap = opt_config.get("max_drawdown_pct_cap")
    min_win_rate_pct = opt_config.get("min_win_rate_pct")
    consistency_subperiods = opt_config.get("consistency_subperiods", 0)
    consistency_weight = opt_config.get("consistency_weight", 0.0)
    consistency_mode = opt_config.get("consistency_mode", "maximin")
    param_stability_penalty_weight = opt_config.get("param_stability_penalty_weight", 0.0)
    
    # Calculate min_test_trades: if explicitly set, use it; otherwise auto-scale from min_trades
    min_test_trades = opt_config.get("min_test_trades")
    if min_test_trades is None:
        # Auto-scale based on period length ratio
        min_test_trades = int(min_trades * test_months / train_months)
        log.info(f"Auto-calculated min_test_trades: {min_test_trades} (from min_trades={min_trades}, test_months={test_months}, train_months={train_months})")
    else:
        log.info(f"Using configured min_test_trades: {min_test_trades}")
    
    # Extract parameter ranges (optional - uses defaults if not provided)
    parameter_ranges = wfo_config.get("parameters", {})
    
    # Generate windows
    data_start = data_df_with_indicators.index[0]
    data_end = data_df_with_indicators.index[-1]
    
    windows = generate_rolling_windows(data_start, data_end, train_months, test_months, step_months)
    
    if not windows:
        raise ValueError("No valid windows generated. Check data range and window sizes.")
    
    # Calculate expected number of periods for validation
    # Formula: (data_end - data_start - train_months - test_months) / step_months + 1
    data_range_months = (data_end - data_start).days / 30.44  # Approximate months
    min_required_months = train_months + test_months
    expected_periods = max(1, int((data_range_months - train_months - test_months) / step_months) + 1)
    
    log.info(f"Data range: {data_start.date()} to {data_end.date()} (~{data_range_months:.1f} months)")
    log.info(f"Generated {len(windows)} rolling windows (expected: ~{expected_periods})")
    
    if len(windows) < expected_periods:
        log.warning(
            f"Fewer periods generated than expected. "
            f"Expected ~{expected_periods} periods, got {len(windows)}. "
            f"Minimum required data range for {expected_periods} periods: "
            f"{train_months + test_months + (expected_periods - 1) * step_months:.1f} months"
        )
    
    if len(windows) == 1:
        log.warning(
            f"Only 1 period will be processed. "
            f"For multiple periods, need at least {train_months + test_months + step_months} months of data. "
            f"Current data range: ~{data_range_months:.1f} months"
        )
    else:
        log.info(f"Successfully generated {len(windows)} periods for walk-forward optimization")
    
    # Get initial capital
    initial_capital = base_config.get("portfolio", {}).get("initial_capital", 10000)
    
    # Process each window
    results_list = []
    
    for period_idx, (train_start, train_end, test_start, test_end) in enumerate(windows):
        log.info(f"\n{'='*60}")
        log.info(f"Processing period {period_idx + 1}/{len(windows)}")
        log.info(f"Train: {train_start.date()} to {train_end.date()}")
        log.info(f"Test:  {test_start.date()} to {test_end.date()}")
        log.info(f"{'='*60}")
        
        # Get warmup requirement for indicator calculation
        indicator_warmup = base_config.get("strategy", {}).get("params", {}).get("indicator_warmup_bars", 200)
        ema_periods = base_config.get("strategy", {}).get("params", {}).get("ema_periods", [8, 200])
        max_ema = max(ema_periods) if ema_periods else 200
        # Use max warmup to ensure all indicators have sufficient history
        # Respect config values - no hardcoded minimum
        warmup_bars = max(indicator_warmup, max_ema)
        
        # Find warmup start date by going backwards warmup_bars positions from period start
        # This ensures we get exactly the required number of bars
        # Find index position of train_start (or closest after if exact match not found)
        train_start_idx = data_df_with_indicators.index.get_indexer([train_start], method='pad')[0]
        
        # Ensure we have enough data before train_start
        if train_start_idx >= warmup_bars:
            warmup_start_idx = train_start_idx - warmup_bars
            train_start_with_warmup = data_df_with_indicators.index[warmup_start_idx]
        else:
            # Not enough data before train_start, use data_start
            log.warning(f"Not enough data for {warmup_bars} bars before {train_start}, using available data")
            train_start_with_warmup = data_df_with_indicators.index[0]
        
        # Slice training data with warmup (indicators will be recalculated in optimizer)
        train_df = data_df_with_indicators[
            (data_df_with_indicators.index >= train_start_with_warmup) &
            (data_df_with_indicators.index < train_end)
        ].copy()
        
        # Find warmup start date for test period
        test_start_idx = data_df_with_indicators.index.get_indexer([test_start], method='pad')[0]
        
        if test_start_idx >= warmup_bars:
            warmup_start_idx = test_start_idx - warmup_bars
            test_start_with_warmup = data_df_with_indicators.index[warmup_start_idx]
        else:
            log.warning(f"Not enough data for {warmup_bars} bars before {test_start}, using available data")
            test_start_with_warmup = data_df_with_indicators.index[0]
        
        # Slice test data with warmup (indicators will be recalculated in validation)
        test_df = data_df_with_indicators[
            (data_df_with_indicators.index >= test_start_with_warmup) &
            (data_df_with_indicators.index < test_end)
        ].copy()
        
        if train_df.empty or test_df.empty:
            log.warning(f"Period {period_idx + 1}: Empty train or test data, skipping")
            continue
        
        # Optimize on training data
        previous_best_params = results_list[-1]["best_params"] if period_idx > 0 and len(results_list) > 0 else None
        try:
            best_params, best_value, study = optimize_period(
                base_config=base_config,
                train_df=train_df,
                objective_metric=objective_metric,
                n_trials=n_trials,
                min_trades=min_trades,
                max_trades=max_trades,
                parameter_ranges=parameter_ranges,
                max_drawdown_pct_cap=max_drawdown_pct_cap,
                min_win_rate_pct=min_win_rate_pct,
                train_start=train_start,
                train_end=train_end,
                consistency_subperiods=consistency_subperiods,
                consistency_weight=consistency_weight,
                consistency_mode=consistency_mode,
                previous_best_params=previous_best_params,
                param_stability_penalty_weight=param_stability_penalty_weight,
            )
        except Exception as e:
            log.error(f"Period {period_idx + 1}: Optimization failed: {e}")
            continue
        
        # Create config with best parameters
        config_with_params = base_config.copy()
        config_with_params["strategy"]["params"].update(best_params)
        
        # Validate on test data
        test_metrics = validate_params(
            config_with_params=config_with_params,
            test_df=test_df,
            date_from=test_start,
            date_to=test_end,
        )
        
        if test_metrics is None:
            log.warning(f"Period {period_idx + 1}: Validation failed")
            continue
        
        # Check test period trade count
        test_num_trades = test_metrics.get("num_trades", 0)
        test_trades_insufficient = False
        if test_num_trades < min_test_trades:
            test_trades_insufficient = True
            log.warning(
                f"Period {period_idx + 1}: Insufficient test trades. "
                f"Got {test_num_trades} trades, required {min_test_trades} minimum. "
                f"Test period: {test_start.date()} to {test_end.date()}"
            )
        
        # Store results
        result = {
            "period_idx": period_idx,
            "train_start": train_start,
            "train_end": train_end,
            "test_start": test_start,
            "test_end": test_end,
            "best_params": best_params,
            "best_value": best_value,
            "test_metrics": test_metrics,
            "test_trades_insufficient": test_trades_insufficient,
            "min_test_trades": min_test_trades,
        }
        
        results_list.append(result)
        
        status_msg = f"Period {period_idx + 1} complete. Test return: {test_metrics.get('return_pct', 0):.2f}%, Trades: {test_num_trades}"
        if test_trades_insufficient:
            status_msg += f" (WARNING: below minimum of {min_test_trades})"
        log.info(status_msg)
    
    # Build combined OOS equity curve
    combined_equity_curve = build_combined_oos_equity_curve(results_list, initial_capital)
    
    # Aggregate results
    summary = aggregate_wfo_results(results_list)

    # Best parameters per metric and best period by objective
    best_params_per_metric = find_best_params_per_metric(results_list)
    best_period_by_objective = find_best_period_by_objective(results_list, objective_metric)
    
    # Calculate combined OOS metrics from equity curve
    if not combined_equity_curve.empty:
        final_value = combined_equity_curve["portfolio_value"].iloc[-1]
        combined_return_pct = ((final_value - initial_capital) / initial_capital) * 100
        
        # Calculate max drawdown
        cummax = combined_equity_curve["portfolio_value"].cummax()
        dd = (combined_equity_curve["portfolio_value"] / cummax) - 1
        combined_max_dd_pct = dd.min() * 100
        
        # Calculate Sharpe (simplified - using portfolio returns)
        returns = combined_equity_curve["portfolio_value"].pct_change().dropna()
        if len(returns) > 0 and returns.std() > 0:
            # Approximate annualization (assuming daily data)
            combined_sharpe = float((returns.mean() / returns.std()) * math.sqrt(252))
        else:
            combined_sharpe = None
        
        summary["combined_oos_total_return_pct"] = combined_return_pct
        summary["combined_oos_max_drawdown_pct"] = combined_max_dd_pct
        summary["combined_oos_sharpe"] = combined_sharpe
    
    return {
        "results_list": results_list,
        "summary": summary,
        "combined_equity_curve": combined_equity_curve,
        "best_params_per_metric": best_params_per_metric,
        "best_period_by_objective": best_period_by_objective,
    }
