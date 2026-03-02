# SPY_INTRADAY_TRENDOSC

SPY 15-min backtester with ATR-based stop/target exits. Uses HiLoATRBands strategy: long entries when HiLoPRO slow %D >= threshold and price touches ATR bands lower; short entries when HiLoPRO slow %D <= threshold and price touches ATR bands upper. Indicators: EMA, ATR, ATR Bands (Charles Schwab style), HiLoPRO.

## Setup

Create a virtual environment and install dependencies:

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r config/requirements.txt
```

### Verifying Setup

After setup, verify everything is installed correctly:

```bash
source venv/bin/activate  # Activate venv
pip check                  # Check for dependency conflicts
python -c "import pandas, optuna, yaml, matplotlib; print('All imports OK')"
```

### Minimal Dependencies

If you only need the backtester (no DBN resampling or WFO), you can install minimal dependencies:

```bash
pip install pandas pyyaml pyarrow matplotlib
```

However, for full functionality including walk-forward optimization, use `config/requirements.txt`.

## Data

Put data at `data/spy_15min_session.parquet` (or set `data.local_path` in `config/V01.yaml`). Only `local_parquet` is supported.

**Building session-only data from Databento 1m:** If you have a resample script (e.g. to 5m or 15m), you can resample a DBN file to regular-session Parquet. Then set `data.local_path` to the output path (e.g. `data/spy_15min_session.parquet`) in `config/V01.yaml`.

**Resampling to higher timeframes (e.g. 30m, 1h for HTF overlay):** From the project root, run (uses project venv; creates it if missing):

```bash
./run_resample.sh
# or with options:
./run_resample.sh --input data/spy_15min_session.parquet --output-dir data
```

Or with venv already activated: `python3 -m data.resample_to_timeframes [--input PATH] [--output-dir DIR]`

Defaults: input from `data.local_path` in `config/V01.yaml` (or `data/spy_15min_session.parquet`), output directory = same as input. Writes `{base}_15min.parquet`, `{base}_30min.parquet`, `{base}_1h.parquet` (useful when base is 15m and you need 30min/1h for HTF TMO).

Required parquet format:

- Columns (case-insensitive): `open`, `high`, `low`, `close`, `volume`
- Datetime: index must be datetime, or include a `datetime` column (it will be set as index)

## Indicators

- **EMA**: Computed for periods in `ema_periods` and `pullback_ema_period`
- **ATR**: Wilder/simple/exponential smoothing; used for stops and targets
- **ATR Bands**: MA center with upper/lower bands at center +/- factor * ATR (Charles Schwab style). Config: `atr_bands_enabled`, `atr_bands_length`, `atr_bands_factor`, etc.

## Exits and session close

Exits are ATR-based (stop/target). You can optionally close any open trade at a fixed time (e.g. market close):

- **`exit_at_session_close`** (bool, default `false`): if `true`, any position still open at the session-close bar is closed at that bar's close price.
- **`session_close_time`** (str, default `"15:55"`): time of the session-close bar, as `HH:MM` or `HH:MM:SS`.

Session-close time is compared to the **index time as-is**. If your data index is in exchange time (e.g. US/Eastern for SPY), use `"15:55"`. If the index is in UTC, use the equivalent UTC time (e.g. `"20:55"` for 15:55 Eastern).

Each exit is tagged with **`exit_reason`**: `stop`, `target`, `session_close`, or `max_hold`. Metrics include counts and percentage by reason.

- **`max_trades_per_session`** (int or null, default null): Max completed trades per calendar day. Use `1` for at most one trade per day; omit or `null` for no cap.

- **`allow_exit_on_entry_bar`** (bool, default `true`): If `false`, stop/target (and breakeven) are not evaluated on the entry bar (avoids same-bar stop-outs); session_close can still apply on the entry bar.

- **`entry_price_previous_close`** (bool): When `true`, the exit module may use the previous bar's close for entry price (pullback logic). When `false`, entry price is the signal bar's close.

When **`atr_multiplier_breakeven`** is set, after price reaches that ATR distance in profit, the stop moves to entry (breakeven).

## Indicator warm-up

- **`indicator_warmup_bars`** (int, default `0`): Number of bars at the start of the series during which no entries are allowed (signals are zeroed). Use this so the first possible entry happens only after indicators have enough history (e.g. 200 for a 200-period EMA). Set to `0` for no warm-up.

## Entry time window

- **`entry_after_time`** (str or null, default null): Entry allowed only at or after this time. Format `HH:MM` or `HH:MM:SS`. Omit or `null` = no lower bound.
- **`entry_before_time`** (str or null, default null): Entry allowed only before this time (exclusive). Format `HH:MM` or `HH:MM:SS`. Omit or `null` = no upper bound.

Valid entries are when bar time is in **[entry_after_time, entry_before_time)** (inclusive lower, exclusive upper). Times are compared to the index time as-is.

## Run

From this folder (with venv activated):

```bash
python backtester.py
```

**Options:**

- `--config PATH` — Config YAML path (default: `config/V01.yaml`)
- `-v`, `--verbose` — Enable debug logging

Results go to `output/backtest_results.csv` (bar-level data with indicators and exit_reason), `output/trades.csv` (per-trade rows with entry/exit price, shares, dollar P&L, etc.), `output/metrics.csv`, and `output/equity_curve.png`. The metrics CSV includes performance metrics plus config params and run identifiers.
