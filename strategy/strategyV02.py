# strategy/strategyV02.py
"""
EMA-touch + TMO turn + ST Trend Oscillator entry strategy.

Long:
- Low of 15m bar touches Base EMA (ema_<base_ema_period>)
- TMO turn from bearish to bullish (oscillator crosses above its EMA) on current bar;
  only the first such turn since the last same-day EMA touch
- ST Trend Oscillator filters:
  - st_trend_oscillator_sim > st_trend_ema_sim on current bar
  - st_trend_oscillator > st_trend_ema on each of the previous
    st_trend_oscillator_bars_above bars
  - st_trend_oscillator_sim - st_trend_ema_sim >= st_trend_oscillator_sim_min_spread

Short: inverse conditions (TMO turn from bullish to bearish; first since touch).
Entry is on the close of the 15m bar where conditions are satisfied.
"""

import pandas as pd


class HiLoATRBands:
    """EMA-touch + TMO turn + ST Trend Oscillator entry strategy."""

    def __init__(self, config=None):
        """
        Initialize strategy from config.
        Reads:
        - base_ema_period
        - st_trend_oscillator_bars_above
        - st_trend_oscillator_sim_min_spread
        - ema_touch_lookback_bars
        - allow_short
        """
        self.config = config or {}
        strategy_cfg = self.config.get("strategy", {}) if self.config else {}
        params = strategy_cfg.get("params", {})

        self.base_ema_period = int(params.get("base_ema_period", 50))
        self.st_bars_above = int(params.get("st_trend_oscillator_bars_above", 10))
        self.st_min_spread = float(params.get("st_trend_oscillator_sim_min_spread", 2.0))
        self.ema_touch_lookback = int(params.get("ema_touch_lookback_bars", 8))
        self.allow_short = bool(strategy_cfg.get("allow_short", True))

        print(
            f"Strategy HiLoATRBands: EMA-touch/TMO turn/ST Trend entry "
            f"(base_ema_period={self.base_ema_period}, "
            f"st_bars_above={self.st_bars_above}, "
            f"st_min_spread={self.st_min_spread}, "
            f"ema_touch_lookback={self.ema_touch_lookback}, "
            f"allow_short={self.allow_short})"
        )

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate long/short entry signals based on:
        - EMA touch (low/high vs ema_<base_ema_period>)
        - TMO turn (long: oscillator crosses above EMA; short: crosses below EMA;
          first turn since last same-day EMA touch only)
        - ST Trend Oscillator filters on current and previous bars.
        """
        if df is None or len(df) == 0:
            raise ValueError("DataFrame is empty or None")

        df = df.copy()

        ema_col = f"ema_{self.base_ema_period}"
        required = [
            "low",
            "high",
            "close",
            ema_col,
            "tmo_main",
            "tmo_signal",
            "st_trend_oscillator",
            "st_trend_ema",
            "st_trend_oscillator_sim",
            "st_trend_ema_sim",
        ]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns for strategy: {missing}")

        # Warmup: need enough history for EMA, ST Trend lookback, and exits' lookback.
        exit_lookback_bars = int(
            self.config.get("strategy", {})
            .get("params", {})
            .get("exit_lookback_bars", 10)
        )
        warmup_bars = max(self.base_ema_period, self.st_bars_above, exit_lookback_bars, 1)

        ema = df[ema_col]
        tmo_main = df["tmo_main"]
        tmo_signal = df["tmo_signal"]
        osc = df["st_trend_oscillator"]
        osc_ema = df["st_trend_ema"]
        osc_sim = df["st_trend_oscillator_sim"]
        osc_ema_sim = df["st_trend_ema_sim"]

        # TMO turn on base timeframe: main crosses above/below signal (replaces ST Trend sim for trigger)
        tmo_bullish_turn = (
            (tmo_main > tmo_signal) & (tmo_main.shift(1) <= tmo_signal.shift(1))
        ).fillna(False).astype(bool)
        tmo_bearish_turn = (
            (tmo_main < tmo_signal) & (tmo_main.shift(1) >= tmo_signal.shift(1))
        ).fillna(False).astype(bool)

        # EMA touch on individual bars
        ema_touch_long = df["low"] <= ema
        ema_touch_short = df["high"] >= ema

        # Require that EMA touch occurred in the previous N bars (per day), not necessarily on the entry bar
        if self.ema_touch_lookback > 0:
            dates = pd.to_datetime(df.index).normalize()

            def _rolling_any(s: pd.Series) -> pd.Series:
                return s.rolling(self.ema_touch_lookback, min_periods=1).max()

            touched_long_recent = (
                ema_touch_long.groupby(dates)
                .apply(_rolling_any)
                .reset_index(level=0, drop=True)
                .shift(1)
                .fillna(False)
                .astype(bool)
            )
            touched_short_recent = (
                ema_touch_short.groupby(dates)
                .apply(_rolling_any)
                .reset_index(level=0, drop=True)
                .shift(1)
                .fillna(False)
                .astype(bool)
            )
        else:
            touched_long_recent = ema_touch_long
            touched_short_recent = ema_touch_short

        dates = pd.to_datetime(df.index).normalize()

        # Last same-day bar where EMA touch occurred (for "first TMO turn since touch")
        def _last_touch_in_day(touch_series: pd.Series) -> pd.Series:
            last = None
            out = {}
            for idx in touch_series.index:
                if touch_series.loc[idx]:
                    last = idx
                out[idx] = last
            return pd.Series(out)

        last_touch_bar_long = (
            ema_touch_long.groupby(dates).apply(_last_touch_in_day).reset_index(level=0, drop=True)
        )
        last_touch_bar_short = (
            ema_touch_short.groupby(dates).apply(_last_touch_in_day).reset_index(level=0, drop=True)
        )

        # First TMO turn of that direction since the last same-day EMA touch
        def _first_turn_since_touch_day(
            day_index: pd.Index,
            last_touch_series: pd.Series,
            turn_ok_series: pd.Series,
        ) -> pd.Series:
            first_turn = pd.Series(False, index=day_index)
            for idx in day_index:
                last_touch = last_touch_series.loc[idx]
                if last_touch is None or (isinstance(last_touch, float) and pd.isna(last_touch)):
                    continue
                mask_between = (day_index > last_touch) & (day_index < idx)
                if mask_between.any() and turn_ok_series.loc[day_index[mask_between]].any():
                    continue
                if turn_ok_series.loc[idx]:
                    first_turn.loc[idx] = True
            return first_turn

        first_tmo_bullish_turn = (
            ema_touch_long.groupby(dates)
            .apply(
                lambda g: _first_turn_since_touch_day(
                    g.index, last_touch_bar_long, tmo_bullish_turn
                )
            )
            .reset_index(level=0, drop=True)
            .astype(bool)
        )
        first_tmo_bearish_turn = (
            ema_touch_short.groupby(dates)
            .apply(
                lambda g: _first_turn_since_touch_day(
                    g.index, last_touch_bar_short, tmo_bearish_turn
                )
            )
            .reset_index(level=0, drop=True)
            .astype(bool)
        )

        # ST Trend filters: previous N bars oscillator vs ema
        above_hist = (osc > osc_ema).astype(float)
        below_hist = (osc < osc_ema).astype(float)
        if self.st_bars_above > 0:
            long_hist_ok = (
                above_hist.rolling(self.st_bars_above, min_periods=self.st_bars_above).min() == 1.0
            )
            short_hist_ok = (
                below_hist.rolling(self.st_bars_above, min_periods=self.st_bars_above).min() == 1.0
            )
        else:
            long_hist_ok = pd.Series(True, index=df.index)
            short_hist_ok = pd.Series(True, index=df.index)

        # Current bar ST Trend relationships
        current_long_st = osc_sim > osc_ema_sim
        current_short_st = osc_sim < osc_ema_sim

        # Sim spread
        spread_long = (osc_sim - osc_ema_sim) >= self.st_min_spread
        spread_short = (osc_ema_sim - osc_sim) >= self.st_min_spread

        # Combine conditions
        long_cond = (
            touched_long_recent
            & (tmo_bullish_turn & first_tmo_bullish_turn)
            & current_long_st
            & long_hist_ok
            & spread_long
        )
        short_cond = (
            touched_short_recent
            & (tmo_bearish_turn & first_tmo_bearish_turn)
            & current_short_st
            & short_hist_ok
            & spread_short
        )

        # Apply warmup: no entries before we have enough history
        if len(df) > warmup_bars:
            valid = pd.Series(False, index=df.index)
            valid.iloc[warmup_bars:] = True
            long_cond &= valid
            short_cond &= valid
        else:
            long_cond[:] = False
            short_cond[:] = False

        if not self.allow_short:
            short_cond[:] = False

        signal = pd.Series(0, index=df.index, dtype=int)

        # Ensure only one direction per bar; prioritize long if both somehow true
        signal[long_cond] = 1
        signal[short_cond & ~long_cond] = -1

        df["signal"] = signal
        df["entry_mode"] = ""
        df.loc[signal == 1, "entry_mode"] = "ema_touch_long"
        df.loc[signal == -1, "entry_mode"] = "ema_touch_short"

        return df
