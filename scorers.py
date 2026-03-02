"""
V01 composite scorers: 5 metrics only (expectancy, max_drawdown_pct, sharpe_ratio, win_rate_pct, trades_per_month).
Used by wfo_optimizer (train composite) and wfo_scorer (test ranking). No pandas dependency.
"""

import math

# Fixed mid-band scores per zone (0-100 scale)
SCORE_BAD = 12
SCORE_AVERAGE = 38
SCORE_GOOD = 63
SCORE_GREAT = 88

# Overall score bands: (min_score, range_label, band_name, description)
SCORE_BANDS = [
    (80, "80+", "Great", "Exceptional; rare/live-viable with high confidence (validate rigorously)."),
    (60, "60–79", "Good", "Solid edge; strong for paper/live testing."),
    (40, "40–59", "Average", "Breakeven/modest; needs refinement (e.g., boost win sizes or reduce drawdowns)."),
    (0, "< 40", "Bad", "Unviable; fix major issues (e.g., negative expectancy, poor recovery)."),
]


def _float(val, default=None):
    if val is None:
        return default
    try:
        f = float(val)
        if math.isnan(f):
            return default
        return f
    except (TypeError, ValueError):
        return default


def score_expectancy(row):
    # Expectancy: $/trade or %/trade. Use better of the two zone outcomes.
    num_trades = _float(row.get("num_trades"), 0) or 1
    total_pnl = _float(row.get("total_dollar_pnl"), 0) or 0
    avg_return_pct = _float(row.get("avg_trade_return_pct"), 0) or 0
    expectancy_dollars = total_pnl / num_trades
    expectancy_pct = avg_return_pct  # already per-trade %

    # Zones: Bad <$10 or <0.05%; Average $10-30 or 0.05-0.15%; Good $30-100+ or 0.15-0.5%; Great >$100 or >0.5%
    score_by_dollar = (
        SCORE_GREAT if expectancy_dollars > 100 else
        SCORE_GOOD if expectancy_dollars >= 30 else
        SCORE_AVERAGE if expectancy_dollars >= 10 else
        SCORE_BAD
    )
    score_by_pct = (
        SCORE_GREAT if expectancy_pct > 0.5 else
        SCORE_GOOD if expectancy_pct >= 0.15 else
        SCORE_AVERAGE if expectancy_pct >= 0.05 else
        SCORE_BAD
    )
    return max(score_by_dollar, score_by_pct), f"${expectancy_dollars:.1f} / {expectancy_pct:.3f}%"


def score_max_drawdown_pct(row):
    # Inverted: lower % = better. Bad >25%; Average 20-25%; Good 10-20%; Great <10-15%
    v = _float(row.get("max_drawdown_pct"))
    if v is None:
        return 0, "N/A"
    v = abs(v)
    if v > 25:
        return SCORE_BAD, f"{v:.1f}%"
    if v > 20:
        return SCORE_AVERAGE, f"{v:.1f}%"
    if v > 10:
        return SCORE_GOOD, f"{v:.1f}%"
    return SCORE_GREAT, f"{v:.1f}%"


def score_sharpe_ratio(row):
    # Bad <0.5-1.0; Average 1.0-1.5; Good 1.5-2.0; Great >2.0
    v = _float(row.get("sharpe_ratio"))
    if v is None:
        return 0, "N/A"
    if v < 1.0:
        return SCORE_BAD, f"{v:.2f}"
    if v < 1.5:
        return SCORE_AVERAGE, f"{v:.2f}"
    if v <= 2.0:
        return SCORE_GOOD, f"{v:.2f}"
    return SCORE_GREAT, f"{v:.2f}"


def score_win_rate_pct(row):
    # Granular: linear ramp 45%->38, 70%->88 so optimizer gets gradient (e.g. 52% > 50% > 48%)
    v = _float(row.get("win_rate_pct"))
    if v is None:
        return 0, "N/A"
    if v < 45:
        return SCORE_BAD, f"{v:.1f}%"
    if v >= 70:
        return SCORE_GREAT, f"{v:.1f}%"
    # 45 <= v < 70: linear interpolation from (45, 38) to (70, 88)
    score = 38 + (88 - 38) * (v - 45) / (70 - 45)
    return score, f"{v:.1f}%"


def score_trades_per_month(row):
    # Score trade frequency: best when inside [min_trades_per_month, max_trades_per_month].
    # row should have num_trades; period from test_period_months (ranking) or train_period_months (optimizer).
    period_months = _float(row.get("test_period_months")) or _float(row.get("train_period_months"))
    if period_months is None or period_months <= 0:
        return SCORE_AVERAGE, "N/A"
    num_trades = _float(row.get("num_trades"), 0) or 0
    trades_per_month = num_trades / period_months
    min_pm = _float(row.get("min_trades_per_month"), 10) or 10
    max_pm = _float(row.get("max_trades_per_month"), 25) or 25
    if min_pm >= max_pm:
        return SCORE_AVERAGE, f"{trades_per_month:.1f}/mo"
    if min_pm <= trades_per_month <= max_pm:
        return SCORE_GREAT, f"{trades_per_month:.1f}/mo"
    if trades_per_month < min_pm:
        # Too few: score degrades as we get further below
        if trades_per_month >= min_pm * 0.5:
            return SCORE_GOOD, f"{trades_per_month:.1f}/mo"
        if trades_per_month >= min_pm * 0.25:
            return SCORE_AVERAGE, f"{trades_per_month:.1f}/mo"
        return SCORE_BAD, f"{trades_per_month:.1f}/mo"
    # Too many (overtrading)
    if trades_per_month <= max_pm * 1.5:
        return SCORE_GOOD, f"{trades_per_month:.1f}/mo"
    if trades_per_month <= max_pm * 2.0:
        return SCORE_AVERAGE, f"{trades_per_month:.1f}/mo"
    return SCORE_BAD, f"{trades_per_month:.1f}/mo"


def score_breakeven_exits_pct(row):
    # Score exit/breakeven mix: sweet spot = some breakeven (locks risk) without over-tightening.
    # Great: 10–25%; Good: 5–10% or 25–40%; Average: 0–5% or 40–55%; Bad: >55%
    num_trades = _float(row.get("num_trades"), 0) or 0
    breakeven_exits = _float(row.get("breakeven_exits"), 0) or 0
    if num_trades <= 0:
        return SCORE_AVERAGE, "N/A"
    breakeven_exits_pct = 100.0 * breakeven_exits / num_trades
    if 10 <= breakeven_exits_pct <= 25:
        return SCORE_GREAT, f"{breakeven_exits_pct:.1f}%"
    if 5 <= breakeven_exits_pct < 10 or 25 < breakeven_exits_pct <= 40:
        return SCORE_GOOD, f"{breakeven_exits_pct:.1f}%"
    if 0 <= breakeven_exits_pct < 5 or 40 < breakeven_exits_pct <= 55:
        return SCORE_AVERAGE, f"{breakeven_exits_pct:.1f}%"
    return SCORE_BAD, f"{breakeven_exits_pct:.1f}%"
