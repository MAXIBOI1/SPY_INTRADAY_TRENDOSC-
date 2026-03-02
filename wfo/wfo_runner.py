# wfo_runner.py
"""
Walk-forward optimization runner.
Main entry point for WFO execution.
"""
import argparse
import json
import logging
import os
import sys
from typing import cast

import matplotlib.pyplot as plt
import pandas as pd
from pandas import Timestamp, Timedelta
import yaml

# Add parent directory to path so we can import from root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtester import compute_all_indicators
from data.data_loader import load_config, fetch_data
from wfo.wfo_framework import run_wfo, calculate_wfo_consistency
from wfo.wfo_scorer import score_all_wfo_periods, rank_wfo_periods

log = logging.getLogger("wfo_runner")


def load_wfo_config(config_path, wfo_config_path=None):
    """
    Load base config and WFO config.
    
    Parameters:
    -----------
    config_path : str
        Path to base YAML config
    wfo_config_path : str | None, optional
        Path to separate WFO config YAML (if None, looks for 'wfo' section in base config)
    
    Returns:
    --------
    tuple
        (base_config: dict, wfo_config: dict)
    """
    base_config = load_config(config_path)
    
    if wfo_config_path:
        wfo_config = load_config(wfo_config_path)
    else:
        # Look for 'wfo' section in base config
        wfo_config = base_config.get("wfo", {})
        if not wfo_config:
            raise ValueError("No WFO config found. Provide --wfo-config or add 'wfo' section to base config.")
    
    return base_config, wfo_config


def save_wfo_results(results, output_dir):
    """
    Save WFO results to files.
    
    Parameters:
    -----------
    results : dict
        Results dict from run_wfo()
    output_dir : str
        Output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    
    results_list = results["results_list"]
    summary = results["summary"]
    combined_equity_curve = results["combined_equity_curve"]
    
    # Get ranking info from ranked periods if available
    ranked_periods = results.get("ranked_periods", [])
    rank_by_period = {p["period_idx"]: p["rank"] for p in ranked_periods}
    score_by_period = {p["period_idx"]: p["score"] for p in ranked_periods}
    score_band_by_period = {p["period_idx"]: p["score_band"] for p in ranked_periods}
    
    # Save per-period results (enhanced with ranking info)
    period_rows = []
    for result in results_list:
        period_idx = result["period_idx"]
        test_metrics = result.get("test_metrics", {})
        
        row = {
            "period": period_idx,
            "rank": rank_by_period.get(period_idx),
            "train_start": result["train_start"],
            "train_end": result["train_end"],
            "test_start": result["test_start"],
            "test_end": result["test_end"],
            "best_value": result["best_value"],
            "train_num_trades": test_metrics.get("num_trades", 0),  # Note: train metrics not stored separately
            "train_return_pct": 0.0,  # Not stored currently
            "test_num_trades": test_metrics.get("num_trades", 0),
            "test_return_pct": test_metrics.get("return_pct", 0.0),
            "test_sharpe": test_metrics.get("sharpe_ratio"),
            "test_calmar": test_metrics.get("calmar_ratio"),
            "test_trades_insufficient": result.get("test_trades_insufficient", False),
            "min_test_trades": result.get("min_test_trades", None),
            "ranking_score": score_by_period.get(period_idx),
            "score_band": score_band_by_period.get(period_idx),
            "best_params_json": json.dumps(result["best_params"]),
        }
        period_rows.append(row)
    
    if period_rows:
        wfo_results_df = pd.DataFrame(period_rows)
        wfo_results_file = os.path.join(output_dir, "wfo_results.csv")
        wfo_results_df.to_csv(wfo_results_file, index=False)
        log.info(f"Saved per-period results to: {wfo_results_file}")
    
    # Save OOS return by year (for consistency reporting)
    oos_by_year = summary.get("oos_return_by_year", [])
    if oos_by_year:
        oos_year_df = pd.DataFrame(oos_by_year)
        oos_year_file = os.path.join(output_dir, "wfo_oos_by_year.csv")
        oos_year_df.to_csv(oos_year_file, index=False)
        log.info(f"Saved OOS return by year to: {oos_year_file}")
    
    # Save summary
    summary_rows = []
    for key, value in summary.items():
        summary_rows.append({"metric": key, "value": value})
    
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_file = os.path.join(output_dir, "wfo_summary.csv")
        summary_df.to_csv(summary_file, index=False)
        log.info(f"Saved summary to: {summary_file}")

    # Save best parameters per metric
    best_params_per_metric = results.get("best_params_per_metric", {})
    if best_params_per_metric:
        best_params_per_metric_file = os.path.join(output_dir, "wfo_best_params_per_metric.json")
        with open(best_params_per_metric_file, "w") as f:
            json.dump(best_params_per_metric, f, indent=2)
        log.info(f"Saved best parameters per metric to: {best_params_per_metric_file}")
    
    # Save combined OOS equity curve
    if not combined_equity_curve.empty:
        equity_curve_file = os.path.join(output_dir, "wfo_combined_oos_equity.csv")
        combined_equity_curve.to_csv(equity_curve_file)
        log.info(f"Saved combined OOS equity curve to: {equity_curve_file}")
        
        # Plot equity curve
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(combined_equity_curve.index, combined_equity_curve["portfolio_value"], label="Combined OOS Strategy")
        ax.set_xlabel("Date")
        ax.set_ylabel("Portfolio Value ($)")
        ax.set_title("Walk-Forward Optimization: Combined Out-of-Sample Equity Curve")
        ax.legend()
        ax.grid(True)
        fig.tight_layout()
        
        equity_curve_png = os.path.join(output_dir, "wfo_combined_oos_equity.png")
        fig.savefig(equity_curve_png)
        plt.close(fig)
        log.info(f"Saved equity curve plot to: {equity_curve_png}")
    
    # Save top 5 strategies (consolidated with best_period_by_objective info)
    ranked_periods = results.get("ranked_periods", [])
    if ranked_periods:
        top_n = 5
        top_strategies = []
        for period in ranked_periods[:top_n]:
            # Create serializable version
            strategy = {
                "rank": period["rank"],
                "period_idx": period["period_idx"],
                "score": period["score"],
                "score_band": period["score_band"],
                "test_start": str(period.get("test_start")) if period.get("test_start") else None,
                "test_end": str(period.get("test_end")) if period.get("test_end") else None,
                "best_params": period.get("best_params", {}),
                "best_value": period.get("best_value"),
            }
            # Add scalar test metrics only
            tm = period.get("test_metrics", {})
            scalar_metrics = {}
            for k, v in tm.items():
                if k.startswith("_"):
                    continue
                if v is None or isinstance(v, (bool, str)):
                    scalar_metrics[k] = v
                elif isinstance(v, (int, float)):
                    scalar_metrics[k] = float(v) if isinstance(v, float) else int(v)
                elif hasattr(v, "item"):
                    scalar_metrics[k] = float(v.item()) if isinstance(v.item(), float) else int(v.item())
                elif hasattr(v, "__float__"):
                    try:
                        scalar_metrics[k] = float(v)
                    except (TypeError, ValueError):
                        pass
            strategy["test_metrics"] = scalar_metrics
            strategy["score_breakdown"] = period.get("score_breakdown", {})
            top_strategies.append(strategy)
        
        # Add best_period_by_objective info if available and different from top-ranked
        best_period_by_objective = results.get("best_period_by_objective", {})
        if best_period_by_objective:
            obj_period_idx = best_period_by_objective.get("period_idx")
            # Check if best_by_objective is already in top strategies
            already_included = any(s["period_idx"] == obj_period_idx for s in top_strategies)
            
            if not already_included:
                # Add best_period_by_objective as additional entry
                tm = best_period_by_objective.get("test_metrics") or {}
                scalar_metrics = {}
                for k, v in tm.items():
                    if k.startswith("_"):
                        continue
                    if v is None or isinstance(v, (bool, str)):
                        scalar_metrics[k] = v
                    elif isinstance(v, (int, float)):
                        scalar_metrics[k] = float(v) if isinstance(v, float) else int(v)
                    elif hasattr(v, "item"):
                        scalar_metrics[k] = float(v.item()) if isinstance(v.item(), float) else int(v.item())
                    elif hasattr(v, "__float__"):
                        try:
                            scalar_metrics[k] = float(v)
                        except (TypeError, ValueError):
                            pass
                
                best_by_obj_strategy = {
                    "rank": None,  # Not ranked, but best by objective
                    "period_idx": obj_period_idx,
                    "score": None,
                    "score_band": "best_by_objective",
                    "test_start": str(best_period_by_objective.get("test_start")) if best_period_by_objective.get("test_start") else None,
                    "test_end": str(best_period_by_objective.get("test_end")) if best_period_by_objective.get("test_end") else None,
                    "best_params": best_period_by_objective.get("params", {}),
                    "best_value": best_period_by_objective.get("best_value"),
                    "test_metrics": scalar_metrics,
                    "score_breakdown": {},
                }
                top_strategies.append(best_by_obj_strategy)
            else:
                # Mark the existing entry as best_by_objective
                for strategy in top_strategies:
                    if strategy["period_idx"] == obj_period_idx:
                        strategy["score_band"] = f"{strategy.get('score_band', '')} (best_by_objective)"
                        break
        
        top_strategies_file = os.path.join(output_dir, "wfo_top_strategies.json")
        with open(top_strategies_file, "w") as f:
            json.dump(top_strategies, f, indent=2)
        log.info(f"Saved top {top_n} strategies to: {top_strategies_file}")


def print_summary_report(results):
    """
    Print summary report to console.
    
    Parameters:
    -----------
    results : dict
        Results dict from run_wfo()
    """
    summary = results["summary"]
    results_list = results["results_list"]
    
    print("\n" + "="*70)
    print("WALK-FORWARD OPTIMIZATION SUMMARY")
    print("="*70)
    
    print(f"\nNumber of periods: {summary.get('num_periods', 0)}")
    
    insufficient_trades_count = summary.get('periods_with_insufficient_test_trades', 0)
    if insufficient_trades_count > 0:
        print(f"  ⚠️  Periods with insufficient test trades: {insufficient_trades_count}")
    
    print("\nOut-of-Sample Performance (across periods):")
    print(f"  Mean return: {summary.get('oos_return_mean', 0):.2f}%")
    print(f"  Std dev:     {summary.get('oos_return_std', 0):.2f}%")
    print(f"  Min:         {summary.get('oos_return_min', 0):.2f}%")
    print(f"  Max:         {summary.get('oos_return_max', 0):.2f}%")
    print(f"  Median:      {summary.get('oos_return_median', 0):.2f}%")
    
    if summary.get("oos_sharpe_mean") is not None:
        print(f"\n  Mean Sharpe:  {summary.get('oos_sharpe_mean', 0):.2f}")
        print(f"  Sharpe std:    {summary.get('oos_sharpe_std', 0):.2f}")
    
    if summary.get("oos_calmar_mean") is not None:
        print(f"\n  Mean Calmar:  {summary.get('oos_calmar_mean', 0):.2f}")
        print(f"  Calmar std:    {summary.get('oos_calmar_std', 0):.2f}")
    
    print(f"\n  Mean Max DD:  {summary.get('oos_max_dd_mean', 0):.2f}%")
    print(f"  Total trades: {summary.get('oos_total_trades', 0)}")
    
    if summary.get("oos_profit_factor_mean") is not None:
        print(f"  Mean PF:       {summary.get('oos_profit_factor_mean', 0):.2f}")
    
    # OOS return by period
    oos_by_period = summary.get("oos_return_by_period", [])
    if oos_by_period:
        print("\nOOS Return by Period:")
        for row in oos_by_period[:15]:  # First 15 periods
            print(f"  Period {row['period_idx']}: {row['test_start'][:10]} to {row['test_end'][:10]}  →  {row['return_pct']:.2f}%")
        if len(oos_by_period) > 15:
            print(f"  ... and {len(oos_by_period) - 15} more periods")
    
    # OOS return by year
    oos_by_year = summary.get("oos_return_by_year", [])
    if oos_by_year:
        print("\nOOS Return by Year (mean return per year):")
        for row in oos_by_year:
            print(f"  {row['year']}: {row['return_pct']:.2f}%")
    
    # Combined OOS metrics
    if "combined_oos_total_return_pct" in summary:
        print("\nCombined Out-of-Sample Performance (simulated live trading):")
        print(f"  Total return:  {summary.get('combined_oos_total_return_pct', 0):.2f}%")
        print(f"  Max drawdown:  {summary.get('combined_oos_max_drawdown_pct', 0):.2f}%")
        if summary.get("combined_oos_sharpe") is not None:
            print(f"  Sharpe ratio:  {summary.get('combined_oos_sharpe', 0):.2f}")
    
    # Parameter stability
    stability_score = summary.get("param_stability_score", 1.0)
    print(f"\nParameter Stability Score: {stability_score:.3f}")
    if stability_score > 0.8:
        print("  → Parameters are stable across periods (good)")
    elif stability_score > 0.6:
        print("  → Parameters show moderate variation")
    else:
        print("  → Parameters vary significantly (may indicate overfitting)")
    
    # Walk-forward consistency
    consistency_metrics = summary.get("consistency_metrics")
    if consistency_metrics:
        print(f"\nWalk-Forward Consistency Metrics:")
        top_n = consistency_metrics.get("top_n", 5)
        consecutive_count = consistency_metrics.get("top_n_consecutive_count", 0)
        consecutive_ratio = consistency_metrics.get("top_n_consecutive_ratio", 0.0)
        rank_stability = consistency_metrics.get("rank_stability", 0.0)
        
        print(f"  Top {top_n} consecutive appearances: {consecutive_count} ({consecutive_ratio:.1%})")
        print(f"  Average rank change: {rank_stability:.2f}")
        if consecutive_ratio > 0.5:
            print("  → Good consistency: top performers appear consecutively")
        elif consecutive_ratio > 0.3:
            print("  → Moderate consistency")
        else:
            print("  → Low consistency: rankings vary significantly between periods")

    # Best parameters per metric
    best_params_per_metric = results.get("best_params_per_metric", {})
    if best_params_per_metric:
        print("\nBest Parameters Per Metric (period with best OOS value):")
        labels = {
            "return_pct": "Best return %",
            "sharpe_ratio": "Best Sharpe",
            "calmar_ratio": "Best Calmar",
            "profit_factor": "Best profit factor",
            "max_drawdown_pct": "Best max drawdown (least negative)",
        }
        for metric_key, entry in best_params_per_metric.items():
            label = labels.get(metric_key, metric_key)
            val = entry.get("value")
            val_str = f"{val:.2f}%" if metric_key in ("return_pct", "max_drawdown_pct") else f"{val:.2f}"
            print(f"  {label}: period {entry['period_idx']} (value: {val_str})")
            print(f"    params: {entry.get('params', {})}")

    # Best period by objective metric
    best_period_by_objective = results.get("best_period_by_objective", {})
    if best_period_by_objective:
        print("\nBest Period By Objective Metric (highest optimization score):")
        print(f"  Period: {best_period_by_objective.get('period_idx')}")
        print(f"  Best value (objective): {best_period_by_objective.get('best_value', 0):.2f}")
        print(f"  Test window: {best_period_by_objective.get('test_start')} to {best_period_by_objective.get('test_end')}")
        print(f"  Parameters: {best_period_by_objective.get('params', {})}")
    
    # Top-ranked periods by comprehensive score
    ranked_periods = results.get("ranked_periods", [])
    if ranked_periods:
        print("\nTop 5 Ranked Periods (by comprehensive metric score):")
        for i, period in enumerate(ranked_periods[:5], start=1):
            test_metrics = period.get("test_metrics", {})
            score = period["score"]
            score_band = period["score_band"]
            period_idx = period["period_idx"]
            return_pct = test_metrics.get("return_pct", 0.0)
            sharpe = test_metrics.get("sharpe_ratio")
            sharpe_str = f"{sharpe:.2f}" if sharpe is not None else "N/A"
            print(f"  {i}. Period {period_idx}: Score {score:.1f} ({score_band}) | Return: {return_pct:.2f}% | Sharpe: {sharpe_str}")
        
        # Compare top-ranked vs best-by-objective
        if ranked_periods:
            top_ranked = ranked_periods[0]
            best_by_objective = results.get("best_period_by_objective", {})
            if best_by_objective:
                obj_period_idx = best_by_objective.get("period_idx")
                if obj_period_idx != top_ranked["period_idx"]:
                    print(f"\n  Note: Top-ranked period ({top_ranked['period_idx']}) differs from best-by-objective ({obj_period_idx})")
                    print(f"        This suggests the comprehensive score weights multiple factors, not just the optimization objective.")
    
    # Best/worst periods
    if results_list:
        test_returns = [r["test_metrics"].get("return_pct", 0) for r in results_list if r.get("test_metrics")]
        if test_returns:
            best_idx = test_returns.index(max(test_returns))
            worst_idx = test_returns.index(min(test_returns))
            print(f"\nBest period (by return): {best_idx + 1} ({max(test_returns):.2f}%)")
            print(f"Worst period (by return): {worst_idx + 1} ({min(test_returns):.2f}%)")
    
    print("\n" + "="*70)


def main():
    parser = argparse.ArgumentParser(description="Run walk-forward optimization")
    parser.add_argument("--config", type=str, default="config/V01.yaml", help="Path to base config YAML")
    parser.add_argument("--wfo-config", type=str, default=None, help="Path to WFO config YAML (optional, can be in base config)")
    parser.add_argument("--output-dir", type=str, default="output", help="Output directory (default: output)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")
    args = parser.parse_args()
    
    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s %(name)s %(message)s")
    
    # Change to strategy root directory (parent of wfo/)
    strategy_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(strategy_root)
    
    try:
        # Load configs
        log.info("Loading configurations...")
        base_config, wfo_config = load_wfo_config(args.config, args.wfo_config)
        
        # Load full dataset
        log.info("Loading data...")
        df = fetch_data(base_config)
        # Type assertion: fetch_data returns pd.DataFrame
        if not isinstance(df, pd.DataFrame) or df.empty:
            log.error("No data loaded")
            sys.exit(1)
        
        # Explicitly convert index elements to Timestamp for type checking
        # Cast index access to handle DatetimeIndex properly
        index_first = cast(Timestamp, df.index[0])
        index_last = cast(Timestamp, df.index[-1])
        data_start: Timestamp = pd.to_datetime(index_first)
        data_end: Timestamp = pd.to_datetime(index_last)
        
        log.info(f"Loaded {len(df)} bars from {data_start} to {data_end}")
        
        # Log data range information for WFO validation
        # Explicitly handle Timedelta calculation
        time_diff: Timedelta = data_end - data_start
        data_range_months = time_diff.days / 30.44  # Approximate months
        
        train_months = wfo_config.get("train_months", 3)
        test_months = wfo_config.get("test_months", 1)
        min_required_months = train_months + test_months
        
        log.info(f"Data range: {data_start.date()} to {data_end.date()} (~{data_range_months:.1f} months)")
        log.info(f"Minimum required for 1 period: {min_required_months} months (train: {train_months}, test: {test_months})")
        
        if data_range_months < min_required_months:
            log.warning(
                f"Insufficient data range for even 1 period. "
                f"Have ~{data_range_months:.1f} months, need at least {min_required_months} months"
            )
        
        # Calculate all indicators once
        log.info("Calculating indicators (one-time calculation)...")
        df_with_indicators = compute_all_indicators(df, base_config)
        log.info("Indicators calculated")
        
        # Run WFO
        log.info("Starting walk-forward optimization...")
        results = run_wfo(base_config, wfo_config, df_with_indicators)
        
        # Score and rank all periods
        log.info("Scoring periods using comprehensive metric scoring...")
        scored_periods = score_all_wfo_periods(results["results_list"], base_config)
        # Get stability weight from config (default 0.10)
        stability_weight = wfo_config.get("ranking", {}).get("stability_weight", 0.10)
        ranked_periods = rank_wfo_periods(scored_periods, all_results_list=results["results_list"], stability_weight=stability_weight)
        results["ranked_periods"] = ranked_periods
        
        # Calculate consistency metrics
        consistency_top_n = wfo_config.get("ranking", {}).get("consistency_top_n", 5)
        consistency_metrics = calculate_wfo_consistency(ranked_periods, top_n=consistency_top_n)
        results["summary"]["consistency_metrics"] = consistency_metrics
        
        # Add top-ranked period info to summary
        if ranked_periods:
            top_ranked = ranked_periods[0]
            results["summary"]["top_ranked_period_idx"] = top_ranked["period_idx"]
            results["summary"]["top_ranked_score"] = top_ranked["score"]
            results["summary"]["top_ranked_score_band"] = top_ranked["score_band"]
        
        log.info(f"Scored {len(ranked_periods)} periods. Top score: {ranked_periods[0]['score']:.1f} ({ranked_periods[0]['score_band']})" if ranked_periods else "No periods scored")
        
        # Save results
        log.info("Saving results...")
        save_wfo_results(results, args.output_dir)
        
        # Print summary
        print_summary_report(results)
        
        log.info("Walk-forward optimization complete!")
        
    except Exception as e:
        log.error(f"Error: {e}", exc_info=args.verbose)
        sys.exit(1)


if __name__ == "__main__":
    main()
