# backtester.py
"""
Execution model (explicit):
- Entry is the bar where signal flips to non-zero. Entry price is set by the exit module
  (close of signal bar or previous bar when applicable; see entry_price_previous_close and
  entry time window).
- Exit is when position goes to 0 because stop or target was hit; exit price is
  that level (1.5 ATR or 2.5 ATR from entry), not the bar close.
- entry_price, exit_price, trade_return are written by the exit module.
"""
import argparse
import importlib
import logging
import math
import os
import sys

import pandas as pd
import matplotlib.pyplot as plt

# Strategy root = directory containing this script. All paths (data/, output/, config) are relative to it.
_strategy_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _strategy_root)

from data.data_loader import load_config, fetch_data, fetch_htf_data
from indicators.ema import compute_ema
from indicators.atr import compute_atr
from indicators.atr_bands import compute_atr_bands
from indicators.hilopro import compute_hilopro
from indicators.tmo import compute_tmo
from indicators.st_trend_oscillator_pro import compute_st_trend_oscillator_pro

log = logging.getLogger("backtest")


def compute_all_indicators(df, config):
    """
    Compute all indicators on the DataFrame based on config.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with OHLCV data
    config : dict
        Configuration dict with strategy params
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with all indicators added
    """
    df = df.copy()
    params = config.get("strategy", {}).get("params", {})
    
    # Compute EMAs
    ema_periods = params.get("ema_periods", [8, 200])
    for period in ema_periods:
        df = compute_ema(df, period=period)
    
    pullback_ema_period = params.get("pullback_ema_period")
    if pullback_ema_period is not None:
        df = compute_ema(df, period=int(pullback_ema_period))
    
    # Compute ATR
    df = compute_atr(
        df,
        period=params.get("atr_period", 14),
        method=params.get("atr_method", "wilder"),
    )
    
    # Compute ATR bands if enabled
    if params.get("atr_bands_enabled", False):
        df = compute_atr_bands(
            df,
            displace=params.get("atr_bands_displace", 0),
            factor=params.get("atr_bands_factor", 1.0),
            length=params.get("atr_bands_length", 8),
            price=params.get("atr_bands_price", "close"),
            average_type=params.get("atr_bands_average_type", "simple"),
            true_range_average_type=params.get(
                "atr_bands_true_range_average_type", "simple"
            ),
        )
    
    # Compute HiLoPRO
    df = compute_hilopro(df)

    # Higher-timeframe TMO overlay (optional)
    if params.get("htf_tmo_enabled", False):
        timeframe = params.get("htf_tmo_timeframe", "15min")
        allowed_tf = ("15min", "30min", "1h")
        if timeframe not in allowed_tf:
            log.warning(
                "htf_tmo_timeframe must be one of %s, got %r; skipping HTF TMO",
                allowed_tf,
                timeframe,
            )
        else:
            date_from = df.index.min() if len(df) else None
            date_to = df.index.max() if len(df) else None
            htf_df = fetch_htf_data(
                config, timeframe, date_from=date_from, date_to=date_to, base_dir=_strategy_root
            )
            if htf_df is None or len(htf_df) == 0:
                log.warning(
                    "HTF parquet for timeframe %s not found or empty; skipping HTF TMO",
                    timeframe,
                )
            else:
                htf_df = compute_tmo(
                    htf_df,
                    length=params.get("htf_tmo_length", 14),
                    calc_length=params.get("htf_tmo_calc_length", 5),
                    smooth_length=params.get("htf_tmo_smooth_length", 3),
                )
                tmo_cols = [
                    "tmo_main",
                    "tmo_signal",
                    "tmo_bullish",
                    "tmo_bullish_bearish",
                    "tmo_overbought",
                    "tmo_oversold",
                ]
                for col in tmo_cols:
                    if col not in htf_df.columns:
                        continue
                    htf_series = htf_df[col].reindex(df.index, method="ffill")
                    df[f"htf_tmo_{col.replace('tmo_', '', 1)}"] = htf_series
                log.info("HTF TMO overlay applied (timeframe=%s)", timeframe)

    # ST Trend Oscillator PRO (optional)
    if params.get("st_trend_oscillator_enabled", False):
        timeframe = params.get("st_trend_oscillator_timeframe", "15m")
        trend_osc_seed = params.get("st_trend_oscillator_trend_osc_seed", 50)
        ema_seed = params.get("st_trend_oscillator_ema_seed", 50)
        prev_30min_close_seed = params.get("st_trend_oscillator_prev_30min_close_seed")
        kwargs = {
            "timeframe": timeframe,
            "trend_osc_seed": float(trend_osc_seed) if isinstance(trend_osc_seed, (int, float)) else 50.0,
            "ema_seed": float(ema_seed) if isinstance(ema_seed, (int, float)) else 50.0,
        }
        if prev_30min_close_seed is not None and isinstance(prev_30min_close_seed, (int, float)):
            kwargs["prev_30min_close_seed"] = float(prev_30min_close_seed)
        df = compute_st_trend_oscillator_pro(df, **kwargs)
        log.info("ST Trend Oscillator PRO applied (timeframe=%s)", timeframe)

    return df


def _extract_metrics_from_row(metrics_row):
    """
    Extract key metrics from metrics_row dict for return value.
    
    Parameters:
    -----------
    metrics_row : dict
        Metrics dict with all metrics
    
    Returns:
    --------
    dict
        Subset of metrics for programmatic use
    """
    return {
        "num_trades": metrics_row.get("num_trades", 0),
        "avg_trade_return_pct": metrics_row.get("avg_trade_return_pct", 0.0),
        "return_pct": metrics_row.get("return_pct", 0.0),
        "sharpe_ratio": metrics_row.get("sharpe_ratio"),
        "calmar_ratio": metrics_row.get("calmar_ratio"),
        "sortino_ratio": metrics_row.get("sortino_ratio"),
        "max_drawdown_pct": metrics_row.get("max_drawdown_pct", 0.0),
        "profit_factor": metrics_row.get("profit_factor"),
        "realized_rr": metrics_row.get("realized_rr"),
        "win_rate_pct": metrics_row.get("win_rate_pct", 0.0),
        "total_dollar_pnl": metrics_row.get("total_dollar_pnl", 0.0),
        "ending_portfolio": metrics_row.get("ending_portfolio", 0.0),
        "initial_capital": metrics_row.get("initial_capital", 0.0),
        "max_consecutive_losses": metrics_row.get("max_consecutive_losses", 0),
        "avg_trade_duration_bars": metrics_row.get("avg_trade_duration_bars", 0.0),
        "breakeven_exits": metrics_row.get("breakeven_exits", 0),
    }


def _calculate_trade_pnl(
    row,
    df,
    cumulative_capital,
    risk_dollars_per_trade,
    use_fixed_risk,
    risk_pct_per_trade,
    risk_cap_pct,
    stop_mult,
    target_mult,
    dollar_pnl_mode,
    breakeven_tol,
    commission_per_side,
    slippage_bps,
):
    """
    Calculate position size and dollar P&L for a single trade.
    
    Returns:
        tuple: (share_count, dollar_pnl, risk_dollars)
    """
    entry_bar_i = int(row["entry_bar_index"])
    atr_at_entry = float(df["atr"].iloc[entry_bar_i])
    entry_price = float(row["entry_price"])
    exit_price = float(row["exit_price"])
    trade_return = float(row["trade_return"])
    direction = 1 if (exit_price - entry_price) * trade_return >= 0 else -1
    exit_reason = (row.get("exit_reason") or "").strip()

    # Cap risk by capital (both paths) to prevent negative capital
    raw_risk_dollars = risk_dollars_per_trade if use_fixed_risk else cumulative_capital * risk_pct_per_trade
    if risk_cap_pct > 0:
        risk_dollars = min(raw_risk_dollars, max(0.0, cumulative_capital) * risk_cap_pct)
    else:
        risk_dollars = raw_risk_dollars

    # Use band levels when available (ATRBandsExit); else ATR multipliers
    row_stop = row.get("stop_level", float("nan"))
    row_target = row.get("target_level", float("nan"))
    use_bands = (
        pd.notna(row_stop) and pd.notna(row_target)
        and math.isfinite(row_stop) and math.isfinite(row_target)
    )
    if use_bands:
        row_stop = float(row_stop)
        row_target = float(row_target)
        risk_per_share = abs(entry_price - row_stop)
        risk_dist = abs(entry_price - row_stop)
        target_dist = abs(row_target - entry_price)
        rr_ratio = target_dist / risk_dist if risk_dist > 0 else 0.0
    else:
        risk_per_share = atr_at_entry * stop_mult
        rr_ratio = target_mult / stop_mult if stop_mult else 0.0

    if math.isnan(risk_per_share) or risk_per_share <= 0 or not math.isfinite(risk_per_share):
        share_count = 0
    else:
        share_count = max(1, int(risk_dollars / risk_per_share)) if risk_dollars > 0 else 0

    if dollar_pnl_mode == "options":
        # Path 2: fixed payoff by exit type; breakeven = stop with exit_price ≈ entry_price
        win_dollars = risk_dollars * rr_ratio if risk_dollars > 0 else 0.0
        if risk_dollars <= 0:
            dollar_pnl = 0.0
        elif exit_reason == "target":
            dollar_pnl = win_dollars
        elif exit_reason == "stop":
            if abs(exit_price - entry_price) <= breakeven_tol:
                dollar_pnl = 0.0
            else:
                dollar_pnl = -risk_dollars
        elif exit_reason in ("session_close", "max_hold"):
            # Prorate by distance to stop/target; cap at [-risk_dollars, win_dollars]
            if use_bands:
                stop_level = row_stop
                target_level = row_target
            elif direction == 1:
                stop_level = entry_price - atr_at_entry * stop_mult
                target_level = entry_price + atr_at_entry * target_mult
            else:
                stop_level = entry_price + atr_at_entry * stop_mult
                target_level = entry_price - atr_at_entry * target_mult
            eps = 1e-9
            if direction == 1:
                if exit_price >= target_level:
                    dollar_pnl = win_dollars
                elif exit_price <= stop_level:
                    dollar_pnl = -risk_dollars
                elif exit_price >= entry_price and (target_level - entry_price) > eps:
                    dollar_pnl = win_dollars * (exit_price - entry_price) / (target_level - entry_price)
                elif exit_price <= entry_price and (entry_price - stop_level) > eps:
                    dollar_pnl = -risk_dollars * (entry_price - exit_price) / (entry_price - stop_level)
                else:
                    dollar_pnl = 0.0
            else:
                if exit_price <= target_level:
                    dollar_pnl = win_dollars
                elif exit_price >= stop_level:
                    dollar_pnl = -risk_dollars
                elif exit_price <= entry_price and (entry_price - target_level) > eps:
                    dollar_pnl = win_dollars * (entry_price - exit_price) / (entry_price - target_level)
                elif exit_price >= entry_price and (stop_level - entry_price) > eps:
                    dollar_pnl = -risk_dollars * (exit_price - entry_price) / (stop_level - entry_price)
                else:
                    dollar_pnl = 0.0
            dollar_pnl = max(-risk_dollars, min(win_dollars, dollar_pnl))
        else:
            dollar_pnl = 0.0
    else:
        # Path 1 (equity): shares x (eff_exit - eff_entry) - commissions
        if direction == 1:
            eff_entry = entry_price * (1 + slippage_bps / 10_000)
            eff_exit = exit_price * (1 - slippage_bps / 10_000)
        else:
            eff_entry = entry_price * (1 - slippage_bps / 10_000)
            eff_exit = exit_price * (1 + slippage_bps / 10_000)
        dollar_pnl = share_count * (eff_exit - eff_entry) * direction - 2 * commission_per_side
    
    return share_count, dollar_pnl, risk_dollars


def _generate_entry_strategy_note(config):
    """Generate a human-readable description of the entry strategy from config."""
    return "Stub (no entry signals)"


def _generate_exit_strategy_note(config):
    """Generate a human-readable description of the exit strategy from config."""
    return "Blank slate (session close / max hold only)"


def _cfg_value(value):
    """Serialize a config value for CSV (list/dict/None -> string)."""
    if value is None:
        return ""
    if isinstance(value, list):
        return ",".join(str(x) for x in value)
    if isinstance(value, dict):
        return str(value)
    return value


def run_backtest(config_path=None, config=None, df=None, date_from=None, date_to=None, output_dir=None, output_suffix=None, skip_indicators=False):
    """
    Run backtest with given configuration.
    
    Parameters:
    -----------
    config_path : str | None, optional
        Path to YAML config file (used if config not provided)
    config : dict | None, optional
        Configuration dict (alternative to config_path)
    df : pd.DataFrame | None, optional
        Pre-loaded DataFrame with data (alternative to loading via fetch_data)
    date_from : str | pd.Timestamp | None, optional
        Start date for data filtering (only used if df is None)
    date_to : str | pd.Timestamp | None, optional
        End date for data filtering (only used if df is None)
    output_dir : str | None, optional
        Output directory (defaults to output/)
    output_suffix : str | None, optional
        Suffix to append to output filenames (e.g., "period_1")
    skip_indicators : bool, optional
        If True and df provided, skip indicator computation (assumes already calculated)
    
    Returns:
    --------
    dict | None
        Metrics dict if successful, None if no data/trades (for backward compatibility)
    """
    # Store original config_path for metrics
    original_config_path = config_path
    
    # Handle config loading
    if config is None:
        if config_path is None:
            config_path = os.path.join(_strategy_root, "config", "V01.yaml")
        os.chdir(_strategy_root)  # so data/ and output/ paths in config resolve from strategy root
        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
        log.info("Starting backtest...")
        
        try:
            config = load_config(config_path)
        except FileNotFoundError:
            log.error("Config file not found: %s", config_path)
            sys.exit(1)
        except Exception as e:
            log.error("Error loading config: %s", e)
            sys.exit(1)
    else:
        # If config dict provided, ensure we're in right directory
        if _strategy_root:
            os.chdir(_strategy_root)
        if not logging.getLogger().handlers:
            logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
        # Use provided config_path or empty string
        if original_config_path is None:
            original_config_path = ""

    if not config or "strategy" not in config:
        config_path_str = config_path or "provided config dict"
        raise ValueError(f"Config '{config_path_str}' is empty or missing 'strategy' section")

    strategy_name = config["strategy"]["name"]
    log.info("Using strategy: %s", strategy_name)

    # Handle data loading
    if df is None:
        log.info("Loading data...")
        df = fetch_data(config, date_from=date_from, date_to=date_to)
        log.info("Data loaded: %d rows", len(df))
    else:
        # If df provided, apply date filtering if needed
        if date_from is not None or date_to is not None:
            if date_from is not None:
                if isinstance(date_from, str):
                    date_from = pd.to_datetime(date_from)
                df = df[df.index >= date_from]
            if date_to is not None:
                if isinstance(date_to, str):
                    date_to = pd.to_datetime(date_to)
                df = df[df.index < date_to]
    
    if df is None or len(df) == 0:
        log.warning("No data loaded. Stopping.")
        return None

    # Compute indicators if needed
    if not skip_indicators:
        log.info("Computing indicators (EMAs, ATR, HiLoPRO)...")
        df = compute_all_indicators(df, config)
        log.info("Indicators done.")
    # else: assume indicators already calculated in df

    strategy_module_name = config["strategy"].get("strategy_module", "strategyV02")
    class_name = config["strategy"].get("class_name", "HiLoATRBands")
    module_path = f"strategy.{strategy_module_name}"

    try:
        module = importlib.import_module(module_path)
        strategy_class = getattr(module, class_name)
    except ImportError as e:
        log.error("ImportError loading strategy %s: %s", module_path, e)
        sys.exit(1)
    except AttributeError as e:
        log.error("Strategy module has no class '%s': %s", class_name, e)
        sys.exit(1)

    strategy = strategy_class(config=config)
    df = strategy.generate_signals(df)

    # Dynamic exit class loading (similar to strategy loading)
    exit_module_name = config["strategy"].get("exit_module", "atr_bands_exit")
    exit_class_name = config["strategy"].get("exit_class_name", "ATRBandsExit")
    exit_module_path = f"exits.{exit_module_name}"

    try:
        exit_module = importlib.import_module(exit_module_path)
        exit_class = getattr(exit_module, exit_class_name)
    except ImportError as e:
        log.error("ImportError loading exit %s: %s", exit_module_path, e)
        sys.exit(1)
    except AttributeError as e:
        log.error("Exit module has no class '%s': %s", exit_class_name, e)
        sys.exit(1)

    params = config["strategy"]["params"]
    exit_logic = exit_class(
        exit_at_session_close=params.get("exit_at_session_close", False),
        session_close_time=params.get("session_close_time", "15:55"),
        max_trades_per_session=params.get("max_trades_per_session"),
        allow_exit_on_entry_bar=params.get("allow_exit_on_entry_bar", True),
        max_hold_bars=params.get("max_hold_bars"),
        stop_adjustment_factor=params.get("stop_adjustment_factor", 1.0),
        exit_target_strategy=params.get("exit_target_strategy", "bands"),
        atr_target_multiplier=params.get("atr_multiplier_target", 2.5),
        atr_stop_multiplier=params.get("atr_multiplier_stop", 1.5),
        breakeven_enabled=params.get("breakeven_enabled", True),
    )
    df = exit_logic.apply_exit(df, config=config)

    df["strategy_returns"] = df["position"].shift(1) * df["close"].pct_change()
    df["cum_strategy"] = (1 + df["strategy_returns"].fillna(0)).cumprod()
    df["cum_buy_hold"] = (1 + df["close"].pct_change().fillna(0)).cumprod()

    # Set output directory
    if output_dir is None:
        output_dir = os.path.join(_strategy_root, "output")
    os.makedirs(output_dir, exist_ok=True)
    
    # Handle output suffix
    suffix_str = f"_{output_suffix}" if output_suffix else ""
    output_file = os.path.join(output_dir, f"backtest_results{suffix_str}.csv")
    df.to_csv(output_file)
    log.info("Saved backtest results to: %s", output_file)

    trades = df[df["exit_price"] > 0].copy()

    # Portfolio and costs (configurable)
    initial_capital = config.get("portfolio", {}).get("initial_capital", 100_000)
    commission_per_side = config.get("costs", {}).get("commission_per_side", 0)
    slippage_bps = config.get("costs", {}).get("slippage_bps", 0)
    risk_pct_per_trade = config["strategy"]["params"].get("risk_pct_per_trade", 0.01)
    stop_mult = config["strategy"]["params"].get("atr_multiplier_stop", 1.5)
    target_mult = config["strategy"]["params"].get("atr_multiplier_target", 2.5)
    risk_dollars_per_trade = config["strategy"]["params"].get("risk_dollars_per_trade")
    use_fixed_risk = risk_dollars_per_trade is not None and float(risk_dollars_per_trade) > 0
    if use_fixed_risk:
        risk_dollars_per_trade = float(risk_dollars_per_trade)
        log.info("Position sizing: fixed $%s risk per trade", risk_dollars_per_trade)
    else:
        log.info("Position sizing: %.2f%% of capital per trade", risk_pct_per_trade * 100)

    risk_cap_pct = config["strategy"]["params"].get("risk_cap_pct_of_capital")
    if risk_cap_pct is not None:
        risk_cap_pct = float(risk_cap_pct)
    else:
        risk_cap_pct = 1.0
    dollar_pnl_mode = (config["strategy"]["params"].get("dollar_pnl_mode") or "equity").strip().lower()
    if dollar_pnl_mode not in ("options", "equity"):
        dollar_pnl_mode = "equity"
    breakeven_tol = float(config["strategy"]["params"].get("breakeven_tol", 0.01))
    if dollar_pnl_mode == "options":
        log.info("Dollar P&L: options-style (fixed payoff by exit type); risk capped at %.0f%% of capital", risk_cap_pct * 100)

    # Position sizing and dollar P&L per trade (fixed risk or % of capital at entry; options-style or equity-style P&L)
    trades = trades.sort_index()
    capitals = []
    shares_list = []
    dollar_pnls = []
    cumulative_capital = initial_capital
    
    for idx, row in trades.iterrows():
        share_count, dollar_pnl, _ = _calculate_trade_pnl(
            row=row,
            df=df,
            cumulative_capital=cumulative_capital,
            risk_dollars_per_trade=risk_dollars_per_trade,
            use_fixed_risk=use_fixed_risk,
            risk_pct_per_trade=risk_pct_per_trade,
            risk_cap_pct=risk_cap_pct,
            stop_mult=stop_mult,
            target_mult=target_mult,
            dollar_pnl_mode=dollar_pnl_mode,
            breakeven_tol=breakeven_tol,
            commission_per_side=commission_per_side,
            slippage_bps=slippage_bps,
        )
        capitals.append(cumulative_capital)
        shares_list.append(share_count)
        dollar_pnls.append(dollar_pnl)
        cumulative_capital += dollar_pnl
    if not trades.empty:
        trades["capital_at_entry"] = capitals
        trades["shares"] = shares_list
        trades["dollar_pnl"] = dollar_pnls
    ending_portfolio = initial_capital + sum(dollar_pnls)
    total_commissions = len(trades) * 2 * commission_per_side
    total_slippage = (slippage_bps / 10_000) * (
        (trades["shares"] * trades["entry_price"] + trades["shares"] * trades["exit_price"]).sum()
    ) if not trades.empty and slippage_bps else 0.0
    return_pct = (ending_portfolio - initial_capital) / initial_capital * 100 if initial_capital else 0.0

    trades_file = os.path.join(output_dir, f"trades{suffix_str}.csv")
    trades.to_csv(trades_file)
    log.info("Saved trades to: %s", trades_file)

    # Equity curve: strategy portfolio value (matches metrics) vs buy-and-hold in dollars
    portfolio_curve = pd.Series(index=df.index, dtype=float)
    if not trades.empty:
        for i in range(len(trades)):
            exit_time = trades.index[i]
            portfolio_curve.loc[exit_time] = initial_capital + sum(dollar_pnls[: i + 1])
        portfolio_curve = portfolio_curve.sort_index().ffill()
    portfolio_curve = portfolio_curve.fillna(initial_capital)
    buy_hold_dollars = initial_capital * df["cum_buy_hold"]

    fig, ax = plt.subplots()
    ax.plot(portfolio_curve.index, portfolio_curve.to_numpy(), label="Strategy (sizing & costs)")
    ax.plot(buy_hold_dollars.index, buy_hold_dollars.to_numpy(), label="Buy & Hold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio value ($)")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    equity_curve_path = os.path.join(output_dir, f"equity_curve{suffix_str}.png")
    fig.savefig(equity_curve_path)
    plt.close(fig)
    log.info("Saved equity curve to: %s", equity_curve_path)

    # Best and worst year vs SPY (smaller-scale equity curves)
    years = pd.Series(df.index).dt.year.unique()
    years = years[years != 2026]  # exclude current/partial year
    if len(years) >= 2:
        annual = []
        for y in years:
            mask = df.index.year == y
            if not mask.any():
                continue
            sub = df.loc[mask]
            first_ts = sub.index[0]
            last_ts = sub.index[-1]
            strat_first = portfolio_curve.loc[first_ts]
            strat_last = portfolio_curve.loc[last_ts]
            spy_first = buy_hold_dollars.loc[first_ts]
            spy_last = buy_hold_dollars.loc[last_ts]
            if strat_first <= 0 or spy_first <= 0:
                continue
            strat_ret = (strat_last / strat_first) - 1
            spy_ret = (spy_last / spy_first) - 1
            annual.append((y, strat_ret, spy_ret))
        if len(annual) >= 2:
            annual.sort(key=lambda x: x[1], reverse=True)  # by strategy return desc
            best_year = annual[0][0]
            worst_year = annual[-1][0]
            best_strat_ret = annual[0][1]
            best_spy_ret = annual[0][2]
            worst_strat_ret = annual[-1][1]
            worst_spy_ret = annual[-1][2]
            log.info(
                "Best year: %d (strategy %.2f%%, SPY %.2f%%) | Worst year: %d (strategy %.2f%%, SPY %.2f%%)",
                best_year, best_strat_ret * 100, best_spy_ret * 100,
                worst_year, worst_strat_ret * 100, worst_spy_ret * 100,
            )
            fig2, (ax_best, ax_worst) = plt.subplots(1, 2, figsize=(10, 4))
            for ax, year, title in [(ax_best, best_year, f"Best year ({best_year})"), (ax_worst, worst_year, f"Worst year ({worst_year})")]:
                mask = df.index.year == year
                p_slice = portfolio_curve.loc[mask]
                b_slice = buy_hold_dollars.loc[mask]
                if p_slice.iloc[0] <= 0 or b_slice.iloc[0] <= 0:
                    continue
                strat_rebased = (p_slice / p_slice.iloc[0]) * initial_capital
                spy_rebased = (b_slice / b_slice.iloc[0]) * initial_capital
                ax.plot(strat_rebased.index, strat_rebased.to_numpy(), label="Strategy (sizing & costs)")
                ax.plot(spy_rebased.index, spy_rebased.to_numpy(), label="Buy & Hold")
                ax.set_xlabel("Date")
                ax.set_ylabel("Portfolio value ($)")
                ax.legend()
                ax.grid(True)
                ax.set_title(title)
            fig2.tight_layout()
            best_worst_path = os.path.join(output_dir, f"equity_curve_best_worst_years{suffix_str}.png")
            fig2.savefig(best_worst_path)
            plt.close(fig2)
            log.info("Saved best/worst year equity curves to: %s", best_worst_path)
        else:
            log.info("Skipping best/worst year plot: fewer than 2 full calendar years with valid data.")
    else:
        log.info("Skipping best/worst year plot: backtest spans fewer than 2 calendar years.")

    # Win rate excludes breakeven: wins / (wins + losses)
    num_wins = (trades["trade_return"] > 0).sum() if not trades.empty else 0
    num_losses = (trades["trade_return"] < 0).sum() if not trades.empty else 0
    win_rate = num_wins / (num_wins + num_losses) * 100 if (num_wins + num_losses) > 0 else 0
    num_trades = len(trades)
    avg_return = trades["trade_return"].mean() * 100 if not trades.empty else 0

    # Planned R:R from config
    stop_mult = config["strategy"]["params"].get("atr_multiplier_stop", 1.5)
    target_mult = config["strategy"]["params"].get("atr_multiplier_target", 2.5)
    planned_rr = target_mult / stop_mult if stop_mult else float("inf")

    # Realized R:R = avg winning return / |avg losing return|
    wins = trades["trade_return"][trades["trade_return"] > 0]
    losses = trades["trade_return"][trades["trade_return"] < 0]
    avg_win = wins.mean() if len(wins) else 0.0
    avg_loss = losses.mean() if len(losses) else 0.0
    if avg_loss != 0:
        realized_rr = avg_win / abs(avg_loss)
    else:
        realized_rr = float("inf") if avg_win else 0.0

    # Profit factor = sum(gains) / abs(sum(losses))
    sum_gains = wins.sum() if len(wins) else 0.0
    sum_losses = losses.sum() if len(losses) else 0.0
    profit_factor = sum_gains / abs(sum_losses) if sum_losses != 0 else float("inf")

    # Drawdown from trade PnL portfolio curve (matches equity chart).
    cummax = portfolio_curve.cummax()
    dd = (portfolio_curve / cummax) - 1
    max_drawdown_pct = dd.min() * 100

    # Average drawdown duration (bars): contiguous segments where dd < 0
    underwater = dd < 0
    if underwater.any():
        groups = (underwater != underwater.shift()).cumsum()
        drawdown_lengths = df.loc[underwater].groupby(groups[underwater]).size()
        avg_drawdown_duration_bars = float(drawdown_lengths.mean())
    else:
        avg_drawdown_duration_bars = 0.0

    # Average trade duration (bars)
    avg_trade_duration_bars = float(trades["trade_bars"].mean()) if not trades.empty and "trade_bars" in trades.columns else 0.0

    # Breakeven exits: exit at entry price (tolerance for float)
    if not trades.empty and "entry_price" in trades.columns:
        tol = 1e-9
        breakeven_exits = int((trades["exit_price"] - trades["entry_price"]).abs().lt(tol).sum())
    else:
        breakeven_exits = 0

    # Exit reason counts (stop, target, session_close, max_hold)
    if not trades.empty and "exit_reason" in trades.columns:
        exits_stop = int((trades["exit_reason"] == "stop").sum())
        exits_target = int((trades["exit_reason"] == "target").sum())
        exits_session_close = int((trades["exit_reason"] == "session_close").sum())
        exits_max_hold = int((trades["exit_reason"] == "max_hold").sum())
        session_close_exits_pct = (exits_session_close / num_trades * 100) if num_trades else 0.0
    else:
        exits_stop = exits_target = exits_session_close = exits_max_hold = 0
        session_close_exits_pct = 0.0

    # Max consecutive losses (from trade outcomes)
    if trades.empty or "dollar_pnl" not in trades.columns:
        max_consecutive_losses = 0
    else:
        pnls = trades["dollar_pnl"]
        is_loss = pnls < 0
        max_run = 0
        current = 0
        for is_l in is_loss:
            if is_l:
                current += 1
                max_run = max(max_run, current)
            else:
                current = 0
        max_consecutive_losses = int(max_run)

    # Bar-level returns for Sharpe/Sortino; annualization
    returns = portfolio_curve.pct_change().dropna()
    returns = returns.replace([float("inf"), float("-inf")], float("nan")).dropna()
    try:
        delta = df.index[-1] - df.index[0]
        num_days = max(1, getattr(delta, "days", None) or int(delta.total_seconds() // 86400))
    except (TypeError, AttributeError):
        num_days = 1
    bars_per_year = 252 * (len(df) / num_days) if num_days else 252

    sharpe_ratio = None
    sortino_ratio = None
    if len(returns) > 0 and returns.std() > 0:
        sharpe_ratio = float((returns.mean() / returns.std()) * math.sqrt(bars_per_year))
    if len(returns) > 0:
        downside = returns[returns < 0]
        if len(downside) > 0:
            downside_std = math.sqrt((downside**2).mean())
            if downside_std > 0:
                sortino_ratio = float((returns.mean() / downside_std) * math.sqrt(bars_per_year))

    # Calmar: annualized return / max drawdown
    if num_days > 0 and abs(max_drawdown_pct) >= 1e-9:
        annual_return = (ending_portfolio / initial_capital) ** (252 / num_days) - 1
        calmar_ratio = float(annual_return / (abs(max_drawdown_pct) / 100))
    else:
        calmar_ratio = None

    # Write metrics.csv (single row: metrics + config params + run identifiers)
    metrics_file = os.path.join(output_dir, f"metrics{suffix_str}.csv")

    total_dollar_pnl = sum(dollar_pnls)
    metrics_row = {
        "strategy_name": strategy_name,
        "num_trades": num_trades,
        "initial_capital": initial_capital,
        "ending_portfolio": ending_portfolio,
        "total_dollar_pnl": total_dollar_pnl,
        "return_pct": return_pct,
        "total_commissions": total_commissions,
        "total_slippage": total_slippage,
        "win_rate_pct": win_rate,
        "avg_trade_return_pct": avg_return,
        "planned_rr": planned_rr,
        "realized_rr": realized_rr,
        "profit_factor": profit_factor,
        "max_drawdown_pct": max_drawdown_pct,
        "avg_drawdown_duration_bars": avg_drawdown_duration_bars,
        "avg_trade_duration_bars": avg_trade_duration_bars,
        "breakeven_exits": breakeven_exits,
        "exits_stop": exits_stop,
        "exits_target": exits_target,
        "exits_session_close": exits_session_close,
        "exits_max_hold": exits_max_hold,
        "session_close_exits_pct": session_close_exits_pct,
        "max_consecutive_losses": max_consecutive_losses,
        "sharpe_ratio": sharpe_ratio,
        "sortino_ratio": sortino_ratio,
        "calmar_ratio": calmar_ratio,
        "entry_strategy_note": _generate_entry_strategy_note(config),
        "exit_strategy_note": _generate_exit_strategy_note(config),
    }
    # Run identifiers (Option B)
    metrics_row["cfg_config_path"] = original_config_path or ""
    metrics_row["cfg_ticker"] = config.get("data", {}).get("ticker", "")
    metrics_row["cfg_timeframe"] = config.get("data", {}).get("timeframe", "")
    # Strategy params (cfg_<key>)
    for key, value in config.get("strategy", {}).get("params", {}).items():
        metrics_row[f"cfg_{key}"] = _cfg_value(value)

    pd.DataFrame([metrics_row]).to_csv(metrics_file, index=False)
    log.info("Saved metrics to: %s", metrics_file)
    
    # Extract metrics for return value (before logging, but after trades/portfolio_curve are defined)
    metrics_dict = _extract_metrics_from_row(metrics_row)
    # Also include trades DataFrame reference for WFO (will be added after trades are defined)

    log.info(
        "Backtest complete. Strategy return (on capital): %.2f%% | Buy&Hold: %s | Trades: %d | Win rate: %.1f%% | Avg trade: %.2f%%",
        return_pct,
        f"{df['cum_buy_hold'].iloc[-1]:.2%}",
        num_trades,
        win_rate,
        avg_return,
    )
    log.info(
        "Portfolio: initial $%s | ending $%s | P&L $%.2f (%.2f%%) | commissions $%.2f | slippage $%.2f",
        f"{initial_capital:,.0f}",
        f"{ending_portfolio:,.0f}",
        total_dollar_pnl,
        return_pct,
        total_commissions,
        total_slippage,
    )
    log.info(
        "R:R planned: %.2f | R:R realized: %s | Profit factor: %s | Max DD: %.2f%% | Avg DD duration: %.1f bars | Avg trade duration: %.1f bars | Breakeven exits: %d",
        planned_rr,
        f"{realized_rr:.2f}" if math.isfinite(realized_rr) else "inf",
        f"{profit_factor:.2f}" if math.isfinite(profit_factor) else "inf",
        max_drawdown_pct,
        avg_drawdown_duration_bars,
        avg_trade_duration_bars,
        breakeven_exits,
    )
    log.info(
        "Exit reasons: stop=%d | target=%d | session_close=%d (%.1f%%) | max_hold=%d",
        exits_stop,
        exits_target,
        exits_session_close,
        session_close_exits_pct,
        exits_max_hold,
    )
    if log.isEnabledFor(logging.DEBUG):
        log.debug("Signal counts: %s", df["signal"].value_counts().to_dict())
        log.debug("Position distribution: %s", df["position"].value_counts().to_dict())
    
    # Add trades DataFrame and portfolio curve to metrics dict for WFO
    metrics_dict["_trades_df"] = trades.copy() if not trades.empty else pd.DataFrame()
    metrics_dict["_portfolio_curve"] = portfolio_curve.copy()
    
    # Return metrics dict for programmatic use
    return metrics_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SPY 15m BTD backtest")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML (default: config/V01.yaml)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")
    args = parser.parse_args()
    if args.verbose:
        logging.getLogger("backtest").setLevel(logging.DEBUG)
    run_backtest(args.config)
