# wfo_scorer.py
"""
Score WFO periods for ranking. Uses wfo.optimization.composite_weights from config (V01-aligned);
fallback to V01 default. Same 5 metrics as optimizer (scorers.py).
"""
import logging

from scorers import (
    SCORE_BANDS,
    score_breakeven_exits_pct,
    score_expectancy,
    score_max_drawdown_pct,
    score_sharpe_ratio,
    score_trades_per_month,
    score_win_rate_pct,
)

log = logging.getLogger("wfo_scorer")

# V01 default weights when config has no composite_weights (sum = 1.0)
DEFAULT_COMPOSITE_WEIGHTS = {
    "expectancy": 0.25,
    "max_drawdown_pct": 0.18,
    "sharpe_ratio": 0.20,
    "win_rate_pct": 0.28,
    "trades_per_month": 0.04,
    "breakeven_exits_pct": 0.05,
}

# Map config composite_weights keys to (display_label, scorer_fn)
COMPOSITE_SCORER_REGISTRY = {
    "expectancy": ("Expectancy", score_expectancy),
    "max_drawdown_pct": ("Max Drawdown %", score_max_drawdown_pct),
    "sharpe_ratio": ("Sharpe Ratio", score_sharpe_ratio),
    "win_rate_pct": ("Win Rate %", score_win_rate_pct),
    "trades_per_month": ("Trades/Month", score_trades_per_month),
    "breakeven_exits_pct": ("Breakeven exits %", score_breakeven_exits_pct),
}


def score_wfo_period(test_metrics, base_config):
    """
    Score a single WFO period's test_metrics for ranking.
    Uses wfo.optimization.composite_weights when present (same as optimizer); else V01 default.
    """
    row = dict(test_metrics)
    if "cfg_timeframe" not in row:
        row["cfg_timeframe"] = base_config.get("data", {}).get("timeframe", "15m")
    if "initial_capital" not in row:
        row["initial_capital"] = base_config.get("portfolio", {}).get("initial_capital", 10000)
    
    opt_config = base_config.get("wfo", {}).get("optimization", {})
    if "min_trades_per_month" not in row:
        row["min_trades_per_month"] = opt_config.get("min_trades_per_month", 10)
    if "max_trades_per_month" not in row:
        row["max_trades_per_month"] = opt_config.get("max_trades_per_month", 25)

    composite_weights = opt_config.get("composite_weights")
    if composite_weights and len(composite_weights) > 0:
        weights = composite_weights
    else:
        weights = DEFAULT_COMPOSITE_WEIGHTS

    weighted_sum = 0.0
    total_weight = 0.0
    component_scores = {}
    for key, weight in weights.items():
        weight = float(weight) if weight is not None else 0.0
        if weight <= 0:
            continue
        entry = COMPOSITE_SCORER_REGISTRY.get(key)
        if entry is None:
            continue
        label, fn = entry
        try:
            score, value = fn(row)
        except Exception as e:
            log.warning(f"Error scoring '{label}': {e}")
            score, value = 0, "Error"
        weighted_sum += score * weight
        total_weight += weight
        component_scores[label] = (score, value)

    overall_score = (weighted_sum / total_weight) if total_weight > 0 else 0.0

    score_band = "Bad"
    for min_score, range_label, band_name, description in SCORE_BANDS:
        if overall_score >= min_score:
            score_band = band_name
            break

    score_breakdown = {
        "overall_score": overall_score,
        "score_band": score_band,
        "component_scores": component_scores,
    }
    return overall_score, score_breakdown


def calculate_period_stability_score(period_idx, all_results_list):
    """
    Calculate stability score for a single period based on how similar
    its parameters are to neighboring periods.
    
    Parameters:
    -----------
    period_idx : int
        Index of the period to score
    all_results_list : list
        All WFO results with best_params
    
    Returns:
    --------
    float
        Stability score (0-100), higher = more stable
    """
    if len(all_results_list) <= 1:
        return 50.0  # Neutral if only one period
    
    if period_idx < 0 or period_idx >= len(all_results_list):
        return 50.0
    
    current_params = all_results_list[period_idx].get("best_params", {})
    if not current_params:
        return 50.0
    
    # Compare with neighbors (previous and next periods)
    neighbors = []
    if period_idx > 0:
        neighbors.append(all_results_list[period_idx - 1].get("best_params", {}))
    if period_idx < len(all_results_list) - 1:
        neighbors.append(all_results_list[period_idx + 1].get("best_params", {}))
    
    if not neighbors:
        return 50.0
    
    # Calculate similarity for each parameter
    similarities = []
    for key, current_value in current_params.items():
        neighbor_values = [n.get(key) for n in neighbors if key in n]
        if not neighbor_values:
            continue
        
        if isinstance(current_value, str):
            # Categorical: match = 1.0, mismatch = 0.0
            matches = sum(1 for v in neighbor_values if v == current_value)
            similarity = matches / len(neighbor_values)
        else:
            # Numeric: normalized distance
            try:
                current_float = float(current_value)
                neighbor_floats = [float(v) for v in neighbor_values]
                if not neighbor_floats:
                    continue
                
                # Calculate average distance normalized by range
                distances = [abs(current_float - n) for n in neighbor_floats]
                avg_distance = sum(distances) / len(distances)
                
                # Find range across all periods for this parameter
                all_values = [all_results_list[i].get("best_params", {}).get(key) 
                             for i in range(len(all_results_list))
                             if key in all_results_list[i].get("best_params", {})]
                all_floats = [float(v) for v in all_values if isinstance(v, (int, float))]
                
                if all_floats:
                    param_range = max(all_floats) - min(all_floats)
                    if param_range > 0:
                        similarity = 1.0 - min(avg_distance / param_range, 1.0)
                    else:
                        similarity = 1.0  # All same value
                else:
                    similarity = 0.5  # Neutral
            except (ValueError, TypeError):
                similarity = 0.5  # Neutral
        
        similarities.append(similarity)
    
    if not similarities:
        return 50.0
    
    # Convert to 0-100 scale
    avg_similarity = sum(similarities) / len(similarities)
    stability_score = avg_similarity * 100
    
    return stability_score


def score_all_wfo_periods(results_list, base_config):
    """
    Score all WFO periods.
    
    Parameters:
    -----------
    results_list : list
        List of result dicts from WFO (each has test_metrics)
    base_config : dict
        Base configuration dict
    
    Returns:
    --------
    list
        List of dicts with period info and scores
    """
    scored_periods = []
    
    opt_config = base_config.get("wfo", {}).get("optimization", {})
    default_min_pm = opt_config.get("min_trades_per_month", 10)
    default_max_pm = opt_config.get("max_trades_per_month", 25)
    
    for result in results_list:
        test_metrics = result.get("test_metrics")
        if test_metrics is None:
            continue
        
        row = dict(test_metrics)
        test_start = result.get("test_start")
        test_end = result.get("test_end")
        if test_start is not None and test_end is not None:
            row["test_period_months"] = (test_end - test_start).days / 30.44
        row["min_trades_per_month"] = default_min_pm
        row["max_trades_per_month"] = default_max_pm
        
        try:
            score, score_breakdown = score_wfo_period(row, base_config)
            
            scored_period = {
                "period_idx": result["period_idx"],
                "score": score,
                "score_band": score_breakdown["score_band"],
                "score_breakdown": score_breakdown,
                "test_start": result.get("test_start"),
                "test_end": result.get("test_end"),
                "test_metrics": test_metrics,
                "best_params": result.get("best_params", {}),
                "best_value": result.get("best_value"),
                "test_trades_insufficient": result.get("test_trades_insufficient", False),
            }
            scored_periods.append(scored_period)
        except Exception as e:
            log.warning(f"Error scoring period {result.get('period_idx', 'unknown')}: {e}")
            continue
    
    return scored_periods


def rank_wfo_periods(scored_periods, all_results_list=None, stability_weight=0.10):
    """
    Rank periods by score (descending), optionally weighted by parameter stability.
    
    Parameters:
    -----------
    scored_periods : list
        List of scored period dicts
    all_results_list : list, optional
        All WFO results (needed for stability calculation)
    stability_weight : float, optional
        Weight for stability in final score (0-1). Default 0.10 (10%).
    
    Returns:
    --------
    list
        Ranked list with rank numbers added
    """
    if all_results_list and stability_weight > 0:
        # Calculate stability-adjusted scores
        for scored_period in scored_periods:
            period_idx = scored_period["period_idx"]
            base_score = scored_period["score"]
            stability_score = calculate_period_stability_score(period_idx, all_results_list)
            
            # Blend base score with stability score
            # Higher stability boosts the score
            adjusted_score = (1 - stability_weight) * base_score + stability_weight * stability_score
            scored_period["stability_score"] = stability_score
            scored_period["adjusted_score"] = adjusted_score
            scored_period["original_score"] = base_score
        
        # Sort by adjusted score
        ranked = sorted(scored_periods, key=lambda x: x.get("adjusted_score", x["score"]), reverse=True)
    else:
        # Original behavior: sort by score only
        ranked = sorted(scored_periods, key=lambda x: x["score"], reverse=True)
    
    # Add rank numbers
    for rank, period in enumerate(ranked, start=1):
        period["rank"] = rank
    
    return ranked
