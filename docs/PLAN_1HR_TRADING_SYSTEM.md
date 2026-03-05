# Plan: Building a Solid 1-Hour SPY/SPX Trading System

You already have a 5-min backtester with strategy, exits, WFO, and resampling to 1h. This plan adapts that stack for a **1-hour timeframe** and gives you a clear path from “stuck” to a testable system.

---

## 1. Why 1-Hour Is Different (and How to Use It)

| Aspect | 5–15 min | 1 hour |
|--------|----------|--------|
| Bars per day | ~78 (5m) to ~26 (15m) | ~6.5 (regular session) |
| Noise | High; many false signals | Lower; each bar carries more weight |
| Indicator periods | Short (e.g. 8, 50, 200 on 5m) | Scale up or use fewer, longer periods |
| Best suited for | Scalping, quick pullbacks | Swing-within-day, clear structure, key levels |
| Session rules | Entry windows in minutes | Entry windows in hours (e.g. first 2–3 bars) |

**Takeaway:** On 1h you get fewer, higher-quality signals. Design for **clarity of structure** (trend, key levels, session context) rather than many small triggers.

---

## 2. High-Level Plan (6 Phases)

1. **Data & config** – Ensure 1h data exists and a 1h config runs.
2. **Hypothesis** – Write down one clear, testable edge (e.g. “trend continuation after first pullback”).
3. **Rules** – Turn the edge into concrete entry/exit rules and filters.
4. **Implementation** – Implement in your codebase (strategy + exits + config).
5. **Backtest & metrics** – Run backtester, define “solid” (e.g. Sharpe, drawdown, win rate).
6. **Iterate** – Change one thing at a time and re-test.

---

## 3. Phase 1: Data & Config (Get 1h Running)

**3.1 Build 1h data**

- You already have resampling that produces `*_1h.parquet` from 5m data.
- From project root (with venv active):
  ```bash
  ./run_resample.sh
  ```
- Confirm `data/spy_5min_session_1h.parquet` (or equivalent) exists.

**3.2 Duplicate config for 1h**

- Copy `config/V01.yaml` → `config/V01_1h.yaml`.
- In `V01_1h.yaml` set:
  - `data.local_path`: path to your **1h** parquet (e.g. `data/spy_5min_session_1h.parquet`).
  - `data.timeframe`: `1h`.
- Run the backtester with the **current** strategy on 1h data:
  ```bash
  python backtester.py --config config/V01_1h.yaml
  ```
- Expect it to run; some indicators (e.g. ST Trend Oscillator, HiLo) are tuned for 5m/15m, so results may be poor. The goal here is **no crashes and 1h bars flowing through**.

**3.3 Optional: SPX**

- SPY and SPX move together; SPX is the index (cash-settled options). Your logic can be identical; only data path and ticker change. When ready, add `data/spx_1h.parquet` and a `config/V01_1h_spx.yaml` if you want to backtest SPX separately.

---

## 4. Phase 2: Hypothesis (One Clear Edge)

Pick **one** edge to test first. Examples that fit 1h well:

- **A. Trend continuation**  
  “After the first 1h bar establishes direction and the second 1h bar pulls back to the EMA, the next bar continues the trend more often than random.”

- **B. Range breakout**  
  “First 2–3 1h bars form a range; break of that range in the direction of the daily trend has positive expectancy.”

- **C. Key level + structure**  
  “When price touches the prior day’s high/low or VWAP on a 1h bar and shows a clear rejection candle, the next 1–2 bars tend to continue in the rejection direction.”

- **D. Session structure**  
  “Longs only after 10:30, shorts only after 11:30 (or similar), to avoid the first 1–2 choppy 1h bars.”

Write your hypothesis in one sentence and what “success” means (e.g. “positive expectancy over 2022–2024” or “Sharpe > 0.8 with max DD < 15%”).

---

## 5. Phase 3: Rules (Entry, Exit, Filters)

Turn the hypothesis into rules your backtester can run.

**5.1 Entry (example for trend continuation)**

- **Trend:** e.g. close > EMA(20) on 1h for longs; close < EMA(20) for shorts.
- **Pullback:** e.g. low (long) or high (short) touched EMA in the last 1–2 bars.
- **Trigger:** e.g. close above prior bar high (long) or close below prior bar low (short).
- **Filters:** e.g. only between 10:30 and 14:00; no trade if ATR(14) is in the top 20% of recent ATR (avoid event days).

**5.2 Exit**

- Reuse your ATR-based exits: stop and target as multiples of ATR(14) on 1h (e.g. 1.5 ATR stop, 2–2.5 ATR target), plus optional session close and max hold in bars.
- On 1h, “max_hold_bars: 3” = 3 hours; “session_close” still makes sense to flatten before the close.

**5.3 Position sizing**

- Keep your existing risk-per-trade (e.g. 0.1% or $X per trade); 1h trades are fewer but larger moves, so ATR-based stops are still appropriate.

Document these in a short “Rules” section (bullet list or table) so you can implement them exactly.

---

## 6. Phase 4: Implementation

**6.1 Strategy**

- **Option A – Reuse HiLoATRBands on 1h:**  
  Point config to 1h data and adjust params (see below). Easiest; good to see how current logic behaves on 1h before changing it.

- **Option B – New 1h strategy class:**  
  In `strategy/` add a new module (e.g. `strategy_1h.py`) with a class that implements your Phase 3 rules:
  - Input: DataFrame with OHLCV + required indicators (EMA, ATR, etc.).
  - Output: same signal contract as now (e.g. long/short signal column).
  - Compute only what you need (e.g. EMA, ATR, prior bar high/low); avoid 5m-specific logic (HiLo, ST Trend Oscillator) unless you explicitly want them on 1h.

**6.2 Indicator periods on 1h**

- Rough scaling from 5m → 1h (factor of 12):
  - EMA 8 on 5m → ~2 on 1h (often use 8–21 on 1h for “fast” trend).
  - EMA 50 on 5m → ~4–6 on 1h; 200 on 5m → ~21 on 1h.
- Practical 1h starting set: e.g. EMA 8, 21, 50; ATR 14. Tune later with WFO.

**6.3 Config (1h)**

- In `config/V01_1h.yaml` (or a new `V01_1h_v2.yaml`):
  - `strategy.strategy_module` / `strategy.class_name`: point to your 1h strategy if you created one.
  - `strategy.params`: base_ema_period (e.g. 21), ema_touch_lookback_bars (e.g. 2), atr_period 14, entry/exit and session_close/max_hold_bars.
  - `no_entries_before` / `no_entries_after`: e.g. `"10:30"` and `"15:00"` for 1h.
  - `session_close_time`: e.g. `"15:00"` or `"15:55"` so the last 1h bar doesn’t hold overnight.

**6.4 Exits**

- Your `ATRBandsExit` (or equivalent) can stay; ensure it uses ATR and levels computed on the **same** 1h bars the strategy uses. If your exit module already uses the DataFrame’s ATR and lookback, it should work once the strategy runs on 1h.

---

## 7. Phase 5: Backtest & “Solid” Definition

**7.1 Run**

```bash
python backtester.py --config config/V01_1h.yaml
```

**7.2 Define “solid” up front**

- Example bar: positive expectancy, Sharpe > 0.7, max drawdown < 15%, win rate 40–55%, at least N trades per month so the sample isn’t tiny.
- Put these in a small “criteria” list and check metrics (e.g. `output/metrics.csv`) against it.

**7.3 Sanity checks**

- No entries in the first 1–2 bars if you use warmup or “no_entries_before”.
- Exits (stop/target/session_close) fire as expected (inspect `output/trades.csv` and `backtest_results.csv`).
- Equity curve and drawdowns look plausible (no single-trade spikes unless intended).

---

## 8. Phase 6: Iterate

- Change **one** thing at a time: e.g. EMA period, entry time window, or ATR multiple.
- Re-run backtest and compare metrics. Use WFO later to search parameter space (e.g. `wfo/` with 1h config) once the base logic is stable.
- If the first hypothesis doesn’t meet your “solid” bar, either refine the rules (e.g. add a filter, tighten entry) or switch to another hypothesis (e.g. from trend continuation to range breakout) and repeat from Phase 2.

---

## 9. Quick Reference: 1h vs Your Current 5m Setup

| Item | 5m (current) | 1h (target) |
|------|----------------|-------------|
| Data | `data/spy_5min_session.parquet` | `data/spy_5min_session_1h.parquet` (or similar) |
| Config | `config/V01.yaml` | `config/V01_1h.yaml` |
| Timeframe in config | `5m` | `1h` |
| EMA periods | e.g. 8, 50, 200 | e.g. 8, 21, 50 (fewer bars, shorter periods in “bars”) |
| Entry window | e.g. 10:00–15:00 | e.g. 10:30–15:00 (first 1–2 bars excluded) |
| Session close | 15:45 | 15:00 or 15:55 (last 1h bar) |
| Max hold | null or bars | e.g. 3–5 bars (hours) |
| Strategy | HiLoATRBands (EMA + HiLo + ST Trend) | Same for quick test, or new 1h-specific class |

---

## 10. Next Concrete Steps

1. Run `./run_resample.sh` and confirm 1h parquet exists.
2. Create `config/V01_1h.yaml` with `local_path` → 1h file and `timeframe: 1h`.
3. Run `python backtester.py --config config/V01_1h.yaml` and fix any errors (e.g. missing HTF file for 1h; you may disable HTF TMO for a first 1h-only test).
4. Write down your **one** hypothesis (Phase 2) and **rules** (Phase 3).
5. Either adjust params in `V01_1h.yaml` for the current strategy or add a small 1h strategy class and wire it in config.
6. Re-run backtest, check metrics and trades, then iterate.

Once you have a hypothesis and rules written, the next step is implementing them in your strategy module and config; we can do that incrementally file-by-file if you share your chosen edge and current `V01_1h.yaml`.
