# exits/atr_bands_exit.py
"""
ATR-based exit with lookback stop, 3 ATR target, and 1.5 ATR breakeven.

Entry gate is signal != 0. On entry:
- Long: stop = min(low) of previous exit_lookback_bars bars
- Short: stop = max(high) of previous exit_lookback_bars bars
- Target: entry_price ± atr_target_multiplier * ATR (up for long, down for short)

While in trade:
- If price reaches breakeven threshold (1.5 ATR in favor), effective stop is moved
  to entry_price.
- Each bar checks stop first, then target. If neither hit, optional exit at
  session close or max_hold_bars.

stop_level and target_level columns store the original stop/target at entry
for sizing/metrics; breakeven affects only the effective stop in the bar logic.
"""
import datetime
import math

import pandas as pd


def _bar_date(ts):
    """Return calendar date of bar timestamp (for tz-aware, date in that tz)."""
    date_fn = getattr(ts, "date", None)
    return date_fn() if callable(date_fn) else None


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


class ATRBandsExit:
    """ATR-based exit with lookback stop, ATR target, and breakeven."""

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
        exit_lookback_bars=10,
        atr_multiplier_breakeven=1.5,
    ):
        self.exit_at_session_close = exit_at_session_close
        self._session_close_time = _parse_session_close_time(session_close_time)
        self.max_trades_per_session = max_trades_per_session
        self.allow_exit_on_entry_bar = allow_exit_on_entry_bar
        n = max_hold_bars
        self.max_hold_bars = int(n) if n is not None and int(n) > 0 else None
        # Kept for backtester constructor compatibility
        self.stop_adjustment_factor = float(stop_adjustment_factor)
        self.exit_target_strategy = exit_target_strategy
        self.atr_target_multiplier = float(atr_target_multiplier)
        self.atr_stop_multiplier = float(atr_stop_multiplier)
        self.breakeven_enabled = bool(breakeven_enabled)
        self.exit_lookback_bars = int(exit_lookback_bars) if int(exit_lookback_bars) > 0 else 10
        self.atr_multiplier_breakeven = float(atr_multiplier_breakeven)

    def apply_exit(self, df, config=None):
        """
        Entry gate: can_enter = (signal_i != 0) with ATR-based stop/target/breakeven.
        """
        if df is None or len(df) == 0:
            raise ValueError("DataFrame is empty or None")

        required = ["signal", "entry_mode", "low", "high", "close", "atr"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}. Available: {list(df.columns)}")

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
        direction = 0
        entry_i = 0
        atr_at_entry = float("nan")
        trade_stop_level = float("nan")
        trade_target_level = float("nan")
        breakeven_reached = False

        for i in range(1, len(df)):
            bar_date = _bar_date(df.index[i])
            signal_i = int(df["signal"].iloc[i]) if df["signal"].iloc[i] != 0 else 0
            bar_ts = df.index[i]
            time_getter = getattr(bar_ts, "time", None)
            bar_time = time_getter() if callable(time_getter) else None

            # 1) If in trade: check stop/target/breakeven, then session_close/max_hold
            if in_trade == 1:
                df.at[df.index[i], "position"] = direction
                exit_price_val = None
                exit_reason_val = None

                bar_low = float(df["low"].iloc[i])
                bar_high = float(df["high"].iloc[i])

                # Breakeven activation: once price moves atr_multiplier_breakeven * ATR in favor
                if self.breakeven_enabled and not breakeven_reached and math.isfinite(atr_at_entry):
                    if direction == 1:
                        be_level = entry_price + self.atr_multiplier_breakeven * atr_at_entry
                        if bar_high >= be_level:
                            breakeven_reached = True
                    else:
                        be_level = entry_price - self.atr_multiplier_breakeven * atr_at_entry
                        if bar_low <= be_level:
                            breakeven_reached = True

                # Effective stop: either original stop or breakeven at entry price
                effective_stop = trade_stop_level
                if breakeven_reached:
                    effective_stop = entry_price

                # Stop first, then target
                if math.isfinite(effective_stop) and math.isfinite(trade_target_level):
                    if direction == 1:
                        if bar_low <= effective_stop:
                            exit_price_val = effective_stop
                            exit_reason_val = "stop"
                        elif bar_high >= trade_target_level:
                            exit_price_val = trade_target_level
                            exit_reason_val = "target"
                    else:
                        if bar_high >= effective_stop:
                            exit_price_val = effective_stop
                            exit_reason_val = "stop"
                        elif bar_low <= trade_target_level:
                            exit_price_val = trade_target_level
                            exit_reason_val = "target"

                # If neither stop nor target hit, fall back to max_hold / session_close
                if self.max_hold_bars is not None and (i - entry_i) >= self.max_hold_bars:
                    if exit_price_val is None:
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
                    df.at[df.index[i], "stop_level"] = trade_stop_level
                    df.at[df.index[i], "target_level"] = trade_target_level
                    in_trade = 0

            # 2) Else: check for new signal (blank slate: any non-zero signal)
            else:
                can_enter = signal_i != 0

                if can_enter:
                    entry_price = float(df["close"].iloc[i])
                    atr_at_entry = float(df["atr"].iloc[i])
                    # Require a valid ATR and enough history for stop lookback
                    if not (math.isfinite(atr_at_entry) and atr_at_entry > 0):
                        continue
                    if i <= self.exit_lookback_bars:
                        continue

                    direction = 1 if signal_i == 1 else -1
                    in_trade = 1
                    entry_i = i
                    breakeven_reached = False

                    # Compute stop from previous N bars
                    start = i - self.exit_lookback_bars
                    if direction == 1:
                        trade_stop_level = float(df["low"].iloc[start:i].min())
                        trade_target_level = entry_price + self.atr_target_multiplier * atr_at_entry
                    else:
                        trade_stop_level = float(df["high"].iloc[start:i].max())
                        trade_target_level = entry_price - self.atr_target_multiplier * atr_at_entry

                    df.at[df.index[i], "position"] = direction
                    df.at[df.index[i], "entry_price"] = entry_price

        return df
