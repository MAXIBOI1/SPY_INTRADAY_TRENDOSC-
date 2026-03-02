# exits/atr_bands_exit.py
"""
ATR Bands-based exit for HiLo strategy.
Entry at close of the bar that first touches the band (long when low touches lower, short when high touches upper).
Stop: band_level - (distance_to_center * stop_adjustment_factor) for long, or ATR-based (entry ± atr_stop_multiplier*ATR) when exit_target_strategy is "atr".
Target: atr_bands_upper (long) or atr_bands_lower (short), or ATR-based (entry ± atr_target_multiplier*ATR) when exit_target_strategy is "atr".
Breakeven: when price hits atr_bands_center, move stop to entry; can be disabled via breakeven_enabled=False.
Writes stop_level and target_level on exit rows for position sizing.
"""
import datetime

import numpy as np
import pandas as pd


def _bar_date(ts):
    """Return calendar date of bar timestamp (for tz-aware, date in that tz)."""
    date_fn = getattr(ts, "date", None)
    return date_fn() if callable(date_fn) else None


def _parse_time_optional(s):
    """Parse 'HH:MM' or 'HH:MM:SS' into datetime.time. Return None for None/empty/invalid."""
    if s is None or (isinstance(s, str) and not s.strip()):
        return None
    s = s.strip() if isinstance(s, str) else str(s)
    parts = s.split(":")
    try:
        h = int(parts[0])
        m = int(parts[1]) if len(parts) > 1 else 0
        sec = int(parts[2]) if len(parts) > 2 else 0
    except (ValueError, IndexError):
        return None
    if not (0 <= h <= 23 and 0 <= m <= 59 and 0 <= sec <= 59):
        return None
    return datetime.time(h, m, sec)


def _parse_session_close_time(s):
    """Parse 'HH:MM' or 'HH:MM:SS' into datetime.time. Raises ValueError if invalid."""
    s = (s or "15:55").strip()
    parts = s.split(":")
    try:
        h = int(parts[0])
        m = int(parts[1]) if len(parts) > 1 else 0
        sec = int(parts[2]) if len(parts) > 2 else 0
    except (ValueError, IndexError) as e:
        raise ValueError(f"session_close_time must be HH:MM or HH:MM:SS, got {s!r}") from e
    if not (0 <= h <= 23 and 0 <= m <= 59 and 0 <= sec <= 59):
        raise ValueError(f"session_close_time out of range: {s!r}")
    return datetime.time(h, m, sec)


def _check_ema_trend_filter(df, i, signal_direction):
    """
    Check if EMA trend filter allows entry for given signal direction.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with EMA columns
    i : int
        Bar index to check
    signal_direction : int
        1 for long, -1 for short
    
    Returns:
    --------
    bool
        True if entry is allowed (EMA trend matches signal direction), False otherwise
    """
    ema_8_col = "ema_8"
    ema_200_col = "ema_200"
    
    # Check if EMA columns exist
    if ema_8_col not in df.columns or ema_200_col not in df.columns:
        # If EMAs not available, allow entry (backward compatibility)
        return True
    
    try:
        ema_8 = float(df[ema_8_col].iloc[i])
        ema_200 = float(df[ema_200_col].iloc[i])
        
        # Check for NaN or invalid values
        if pd.isna(ema_8) or pd.isna(ema_200) or not (pd.notna(ema_8) and pd.notna(ema_200)):
            # If EMAs are NaN, skip entry (safer than allowing)
            return False
        
        # For long signals: require ema_8 > ema_200
        if signal_direction == 1:
            return ema_8 > ema_200
        # For short signals: require ema_8 < ema_200
        elif signal_direction == -1:
            return ema_8 < ema_200
        else:
            # Unknown signal direction, don't allow
            return False
    except (ValueError, IndexError, TypeError):
        # If any error accessing EMAs, don't allow entry
        return False


class ATRBandsExit:
    """Exit logic using ATR band levels: entry at band, stop/target/breakeven from bands or ATR when exit_target_strategy is 'atr'."""

    def __init__(
        self,
        exit_at_session_close=False,
        session_close_time="15:55",
        max_trades_per_session=None,
        allow_exit_on_entry_bar=True,
        max_hold_bars=None,
        stop_adjustment_factor=1.0,
        exit_target_strategy="bands",
        atr_target_multiplier=2.5,
        atr_stop_multiplier=1.0,
        breakeven_enabled=True,
    ):
        self.exit_at_session_close = exit_at_session_close
        self._session_close_time = _parse_session_close_time(session_close_time)
        self.max_trades_per_session = max_trades_per_session
        self.allow_exit_on_entry_bar = allow_exit_on_entry_bar
        n = max_hold_bars
        self.max_hold_bars = int(n) if n is not None and int(n) > 0 else None
        self.stop_adjustment_factor = float(stop_adjustment_factor)
        self.exit_target_strategy = exit_target_strategy
        self.atr_target_multiplier = float(atr_target_multiplier)
        self.atr_stop_multiplier = float(atr_stop_multiplier)
        self.breakeven_enabled = bool(breakeven_enabled)

    def apply_exit(self, df, config=None):
        """
        Apply ATR bands-based exits.
        Only processes rows where entry_mode == 'bands' and signal != 0.
        Entry at close of bar that touches band; stop/target from band geometry or from ATR when exit_target_strategy is 'atr'.
        Breakeven when price hits center (only if breakeven_enabled). Writes stop_level and target_level on exit rows.
        """
        if df is None or len(df) == 0:
            raise ValueError("DataFrame is empty or None")

        required = [
            "hilopro_slow_d", "atr_bands_lower", "atr_bands_center", "atr_bands_upper",
            "entry_mode", "signal", "low", "high", "close",
        ]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}. Available: {list(df.columns)}")
        
        # Check for ATR column if using ATR-based target strategies
        if self.exit_target_strategy in ["atr", "hybrid"]:
            if "atr" not in df.columns:
                raise ValueError("ATR column missing — required for 'atr' or 'hybrid' exit_target_strategy")

        df = df.copy()
        df["position"] = 0
        df["entry_price"] = 0.0
        df["exit_price"] = 0.0
        df["trade_return"] = 0.0
        df["trade_bars"] = 0
        df["exit_reason"] = ""
        df["entry_bar_index"] = 0
        df["stop_level"] = float("nan")
        df["target_level"] = float("nan")

        in_trade = 0
        entry_price = 0.0
        stop_level = 0.0
        target_level = 0.0
        breakeven_level = 0.0
        atr_at_entry = 0.0
        direction = 0
        entry_i = 0
        breakeven_triggered = False
        trades_per_date = {}

        params = (config or {}).get("strategy", {}).get("params", {})
        entry_after = _parse_time_optional(params.get("entry_after_time"))
        entry_before = _parse_time_optional(params.get("entry_before_time"))

        for i in range(1, len(df)):
            bar_date = _bar_date(df.index[i])
            signal_i = int(df["signal"].iloc[i]) if df["signal"].iloc[i] != 0 else 0
            entry_mode_i = (df["entry_mode"].iloc[i] or "").strip().lower()
            low_i = float(df["low"].iloc[i])
            high_i = float(df["high"].iloc[i])

            # Check bar time for entry window
            bar_ts = df.index[i]
            time_getter = getattr(bar_ts, "time", None)
            bar_time = time_getter() if callable(time_getter) else None
            in_entry_window = True
            if bar_time is not None and (entry_after is not None or entry_before is not None):
                in_entry_window = (
                    (entry_after is None or bar_time >= entry_after)
                    and (entry_before is None or bar_time < entry_before)
                )

            # 1) If in trade: check exit (stop/target/session_close)
            if in_trade == 1:
                df.at[df.index[i], "position"] = direction
                exit_price_val = None
                exit_reason_val = None

                # Breakeven: when price hits center, stop moves to entry (only if breakeven_enabled)
                if self.breakeven_enabled and not breakeven_triggered and (self.allow_exit_on_entry_bar or i != entry_i):
                    if direction == 1 and high_i >= breakeven_level:
                        breakeven_triggered = True
                    elif direction == -1 and low_i <= breakeven_level:
                        breakeven_triggered = True

                current_stop = entry_price if (self.breakeven_enabled and breakeven_triggered) else stop_level
                check_stop_target = self.allow_exit_on_entry_bar or (i != entry_i)

                if check_stop_target:
                    # Calculate targets based on exit_target_strategy
                    band_target = target_level
                    if self.exit_target_strategy in ["atr", "hybrid"]:
                        if direction == 1:
                            atr_target = entry_price + (atr_at_entry * self.atr_target_multiplier)
                        else:
                            atr_target = entry_price - (atr_at_entry * self.atr_target_multiplier)
                    else:
                        atr_target = None
                    
                    if direction == 1:
                        # Check stop first
                        if low_i <= current_stop:
                            exit_price_val = current_stop
                            exit_reason_val = "stop"
                        # Then check targets
                        elif self.exit_target_strategy == "bands":
                            if high_i >= band_target:
                                exit_price_val = band_target
                                exit_reason_val = "target"
                        elif self.exit_target_strategy == "atr":
                            if high_i >= atr_target:
                                exit_price_val = atr_target
                                exit_reason_val = "target"
                        else:  # hybrid
                            # Exit at whichever target is hit first
                            band_hit = high_i >= band_target
                            atr_hit = high_i >= atr_target
                            if band_hit and atr_hit:
                                # Both hit - use the one that was hit first (closer to entry)
                                exit_price_val = min(band_target, atr_target)
                                exit_reason_val = "target"
                            elif band_hit:
                                exit_price_val = band_target
                                exit_reason_val = "target"
                            elif atr_hit:
                                exit_price_val = atr_target
                                exit_reason_val = "target"
                    else:  # direction == -1
                        # Check stop first
                        if high_i >= current_stop:
                            exit_price_val = current_stop
                            exit_reason_val = "stop"
                        # Then check targets
                        elif self.exit_target_strategy == "bands":
                            if low_i <= band_target:
                                exit_price_val = band_target
                                exit_reason_val = "target"
                        elif self.exit_target_strategy == "atr":
                            if low_i <= atr_target:
                                exit_price_val = atr_target
                                exit_reason_val = "target"
                        else:  # hybrid
                            # Exit at whichever target is hit first
                            band_hit = low_i <= band_target
                            atr_hit = low_i <= atr_target
                            if band_hit and atr_hit:
                                # Both hit - use the one that was hit first (further from entry)
                                exit_price_val = max(band_target, atr_target)
                                exit_reason_val = "target"
                            elif band_hit:
                                exit_price_val = band_target
                                exit_reason_val = "target"
                            elif atr_hit:
                                exit_price_val = atr_target
                                exit_reason_val = "target"

                if exit_price_val is None and self.max_hold_bars is not None and (i - entry_i) >= self.max_hold_bars:
                    exit_price_val = float(df["close"].iloc[i])
                    exit_reason_val = "max_hold"

                if exit_price_val is None and self.exit_at_session_close and bar_time == self._session_close_time:
                    exit_price_val = float(df["close"].iloc[i])
                    exit_reason_val = "session_close"

                if exit_price_val is not None:
                    df.at[df.index[i], "position"] = 0
                    df.at[df.index[i], "entry_price"] = entry_price
                    df.at[df.index[i], "exit_price"] = exit_price_val
                    df.at[df.index[i], "trade_return"] = (exit_price_val / entry_price - 1) * direction
                    df.at[df.index[i], "trade_bars"] = i - entry_i
                    df.at[df.index[i], "exit_reason"] = exit_reason_val or ""
                    df.at[df.index[i], "entry_bar_index"] = entry_i
                    df.at[df.index[i], "stop_level"] = stop_level
                    df.at[df.index[i], "target_level"] = target_level
                    in_trade = 0
                    if bar_date is not None:
                        trades_per_date[bar_date] = trades_per_date.get(bar_date, 0) + 1

            # 2) Else: check for new signal (entry_mode == "bands")
            else:
                can_enter = (
                    signal_i != 0
                    and entry_mode_i == "bands"
                    and in_entry_window
                    and (
                        self.max_trades_per_session is None
                        or self.max_trades_per_session <= 0
                        or bar_date is None
                        or trades_per_date.get(bar_date, 0) < self.max_trades_per_session
                    )
                    and _check_ema_trend_filter(df, i, signal_i)
                )

                if can_enter:
                    entry_price = float(df["close"].iloc[i])

                    if self.exit_target_strategy == "atr":
                        # ATR-based stop and target from entry
                        atr_at_entry = float(df["atr"].iloc[i])
                        if signal_i == 1:
                            stop_level = entry_price - atr_at_entry * self.atr_stop_multiplier
                            target_level = entry_price + atr_at_entry * self.atr_target_multiplier
                            direction = 1
                        else:
                            stop_level = entry_price + atr_at_entry * self.atr_stop_multiplier
                            target_level = entry_price - atr_at_entry * self.atr_target_multiplier
                            direction = -1
                        breakeven_level = entry_price  # unused when breakeven_enabled is False

                        in_trade = 1
                        entry_i = i
                        breakeven_triggered = False
                        df.at[df.index[i], "position"] = direction
                        df.at[df.index[i], "entry_price"] = entry_price
                    else:
                        lower = float(df["atr_bands_lower"].iloc[i])
                        center = float(df["atr_bands_center"].iloc[i])
                        upper = float(df["atr_bands_upper"].iloc[i])

                        if np.isfinite(lower) and np.isfinite(center) and np.isfinite(upper) and lower > 0 and center > 0 and upper > 0:
                            # Entry at close of the bar that first touches the band (not at band level)
                            if signal_i == 1:
                                distance_to_center = center - lower
                                stop_level = lower - (distance_to_center * self.stop_adjustment_factor)
                                target_level = upper
                                breakeven_level = center
                                direction = 1
                            else:
                                distance_to_center = upper - center
                                stop_level = upper + (distance_to_center * self.stop_adjustment_factor)
                                target_level = lower
                                breakeven_level = center
                                direction = -1

                            # Track ATR at entry for hybrid target strategy
                            if self.exit_target_strategy == "hybrid":
                                atr_at_entry = float(df["atr"].iloc[i])
                            else:
                                atr_at_entry = 0.0

                            in_trade = 1
                            entry_i = i
                            breakeven_triggered = False
                            df.at[df.index[i], "position"] = direction
                            df.at[df.index[i], "entry_price"] = entry_price

        return df
