# exits/atr_bands_exit.py
"""
Blank-slate exit: entry gate is signal != 0; exit only at session close or max_hold_bars.
No stop/target/breakeven. stop_level and target_level set to nan on exit rows (backtester
uses ATR-based fallback for position sizing).
"""
import datetime

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
    """Blank-slate exit: entry when signal != 0; exit only at session_close or max_hold_bars."""

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
        # Unused in blank slate; kept for backtester constructor compatibility
        self.stop_adjustment_factor = float(stop_adjustment_factor)
        self.exit_target_strategy = exit_target_strategy
        self.atr_target_multiplier = float(atr_target_multiplier)
        self.atr_stop_multiplier = float(atr_stop_multiplier)
        self.breakeven_enabled = bool(breakeven_enabled)

    def apply_exit(self, df, config=None):
        """
        Blank slate: can_enter = (signal_i != 0). Exit only at session_close or max_hold_bars.
        stop_level and target_level set to nan on exit rows.
        """
        if df is None or len(df) == 0:
            raise ValueError("DataFrame is empty or None")

        required = ["signal", "entry_mode", "low", "high", "close"]
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

        for i in range(1, len(df)):
            bar_date = _bar_date(df.index[i])
            signal_i = int(df["signal"].iloc[i]) if df["signal"].iloc[i] != 0 else 0
            bar_ts = df.index[i]
            time_getter = getattr(bar_ts, "time", None)
            bar_time = time_getter() if callable(time_getter) else None

            # 1) If in trade: check exit (session_close or max_hold only)
            if in_trade == 1:
                df.at[df.index[i], "position"] = direction
                exit_price_val = None
                exit_reason_val = None

                if self.max_hold_bars is not None and (i - entry_i) >= self.max_hold_bars:
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
                    df.at[df.index[i], "stop_level"] = float("nan")
                    df.at[df.index[i], "target_level"] = float("nan")
                    in_trade = 0

            # 2) Else: check for new signal (blank slate: any non-zero signal)
            else:
                can_enter = signal_i != 0

                if can_enter:
                    entry_price = float(df["close"].iloc[i])
                    direction = 1 if signal_i == 1 else -1
                    in_trade = 1
                    entry_i = i
                    df.at[df.index[i], "position"] = direction
                    df.at[df.index[i], "entry_price"] = entry_price

        return df
