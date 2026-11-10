"""
TradePulse - Step 10: Confidence Gating, Signal Rules & Non-Overlapping Backtesting
Author: TradePulse Team (IIT Patna, IMPACT Lab)

Implements the production signal rule and realistic simulation:
1. apply_confidence_gating: Directional signal if max(P_up, P_down) >= tau and is highest class; else 0 (No clear signal).
2. evaluate_gating_grid: Evaluates fixed probability thresholds and quantile coverage cutoffs with day-level block-bootstrap 95% CIs.
3. simulate_non_overlapping_trades: Simulates 15-minute holding periods where active positions block new entries for 3 bars.
4. check_filled_sensitivity: Checks whether results change when synthetic bars (is_filled=1) are removed.
"""

import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ==============================================================================
# 1. Production Signal Rule
# ==============================================================================

def apply_confidence_gating(
    probs: np.ndarray,
    threshold: float,
    classes: List[int] = [-1, 0, 1],
) -> np.ndarray:
    """
    Directional Signal Rule:
    - Output +1 (Up) if P(Up) is the maximum class probability AND P(Up) >= threshold.
    - Output -1 (Down) if P(Down) is the maximum class probability AND P(Down) >= threshold.
    - Otherwise, output 0 ('No Clear Signal'). Neutral class always maps to 0.
    """
    probs = np.asarray(probs, dtype=float)
    n_samples = len(probs)
    signals = np.zeros(n_samples, dtype=int)

    p_down = probs[:, 0]
    p_neut = probs[:, 1]
    p_up = probs[:, 2]

    # Condition for Down: p_down is strictly highest AND >= threshold
    down_mask = (p_down >= threshold) & (p_down > p_neut) & (p_down > p_up)
    signals[down_mask] = -1

    # Condition for Up: p_up is strictly highest AND >= threshold
    up_mask = (p_up >= threshold) & (p_up > p_neut) & (p_up >= p_down)
    signals[up_mask] = 1

    return signals


def apply_quantile_gating(
    probs: np.ndarray,
    top_fraction: float,
) -> Tuple[np.ndarray, float]:
    """
    Selects the top_fraction highest conviction directional bars:
    conviction = max(P(Down), P(Up)).
    Returns (signals, effective_threshold).
    """
    p_dir = np.maximum(probs[:, 0], probs[:, 2])
    cutoff = float(np.quantile(p_dir, 1.0 - top_fraction))
    signals = apply_confidence_gating(probs, threshold=cutoff)
    return signals, cutoff


# ==============================================================================
# 2. Threshold Evaluation Grid with Day-Level Block Bootstrap
# ==============================================================================

def evaluate_threshold_grid(
    session_dates: pd.Series,
    y_true: np.ndarray,
    actual_returns: np.ndarray,
    probs: np.ndarray,
    thresholds: List[float] = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60],
    quantile_cutoffs: List[float] = [0.01, 0.05, 0.10, 0.20],
    cost_levels: List[float] = [0.0005, 0.0008, 0.0012],
    n_bootstraps: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Evaluates a predefined grid of probability thresholds and quantile cutoffs.
    Computes day-level block-bootstrap 95% confidence intervals for net returns.
    """
    rng = np.random.default_rng(seed)
    unique_dates = np.unique(session_dates)
    n_days = len(unique_dates)

    df_base = pd.DataFrame({
        "date": session_dates.values,
        "y_true": y_true,
        "ret": actual_returns,
        "p_down": probs[:, 0],
        "p_neut": probs[:, 1],
        "p_up": probs[:, 2],
    })
    day_groups = [group for _, group in df_base.groupby("date")]

    records = []

    # Combined candidate specifications
    grid_specs = []
    for t in thresholds:
        grid_specs.append(("prob_threshold", t, f"P >= {t:.2f}"))
    for q in quantile_cutoffs:
        grid_specs.append(("quantile_cutoff", q, f"Top {int(q*100)}% Conviction"))

    for kind, val, desc in grid_specs:
        if kind == "prob_threshold":
            signals = apply_confidence_gating(probs, threshold=val)
            eff_thresh = val
        else:
            signals, eff_thresh = apply_quantile_gating(probs, top_fraction=val)

        active_mask = (signals != 0)
        n_active = int(np.sum(active_mask))
        cov_pct = (n_active / len(signals)) * 100.0

        # Frequency metrics
        daily_counts = [int(np.sum(signals[session_dates.values == d] != 0)) for d in unique_dates]
        signals_per_day = float(np.mean(daily_counts))
        zero_signal_days_pct = (sum(1 for c in daily_counts if c == 0) / n_days) * 100.0

        if n_active > 0:
            act_sig = signals[active_mask]
            act_true = y_true[active_mask]
            act_ret = actual_returns[active_mask]

            # Precision & Direction-only accuracy
            prec = float(np.mean(act_sig == act_true) * 100.0)
            dir_true_mask = (act_true != 0)
            if np.sum(dir_true_mask) > 0:
                dir_acc = float(np.mean(np.sign(act_sig[dir_true_mask]) == np.sign(act_true[dir_true_mask])) * 100.0)
            else:
                dir_acc = 0.0

            gross_rets = act_sig * act_ret
            gross_bps = float(np.mean(gross_rets) * 10000.0)

            # Net returns at cost levels
            net_5bp = float(np.mean(gross_rets - 0.0005) * 10000.0)
            net_8bp = float(np.mean(gross_rets - 0.0008) * 10000.0)
            net_12bp = float(np.mean(gross_rets - 0.0012) * 10000.0)
            win_rate_8bp = float(np.mean((gross_rets - 0.0008) > 0) * 100.0)

            # Day-Level Block Bootstrap for 8bp Net Return
            boot_net_8bps = []
            for _ in range(n_bootstraps):
                sampled_idx = rng.choice(n_days, size=n_days, replace=True)
                sample_df = pd.concat([day_groups[i] for i in sampled_idx], ignore_index=True)
                s_p = sample_df[["p_down", "p_neut", "p_up"]].values
                s_ret = sample_df["ret"].values

                if kind == "prob_threshold":
                    s_sig = apply_confidence_gating(s_p, threshold=val)
                else:
                    s_sig, _ = apply_quantile_gating(s_p, top_fraction=val)

                s_act = (s_sig != 0)
                if np.sum(s_act) > 0:
                    b_net = np.mean((s_sig[s_act] * s_ret[s_act]) - 0.0008) * 10000.0
                    boot_net_8bps.append(b_net)

            if len(boot_net_8bps) > 0:
                ci_low = float(np.percentile(boot_net_8bps, 2.5))
                ci_high = float(np.percentile(boot_net_8bps, 97.5))
            else:
                ci_low, ci_high = 0.0, 0.0
        else:
            prec = 0.0
            dir_acc = 0.0
            gross_bps = 0.0
            net_5bp = 0.0
            net_8bp = 0.0
            net_12bp = 0.0
            win_rate_8bp = 0.0
            ci_low, ci_high = 0.0, 0.0

        records.append({
            "rule_type": kind,
            "rule_desc": desc,
            "effective_threshold": round(eff_thresh, 4),
            "n_signals": n_active,
            "coverage_pct": round(cov_pct, 2),
            "signals_per_day": round(signals_per_day, 1),
            "zero_signal_days_pct": round(zero_signal_days_pct, 1),
            "precision_pct": round(prec, 1),
            "direction_accuracy_pct": round(dir_acc, 1),
            "win_rate_8bp_pct": round(win_rate_8bp, 1),
            "gross_return_bps": round(gross_bps, 2),
            "net_5bp_bps": round(net_5bp, 2),
            "net_8bp_bps": round(net_8bp, 2),
            "net_8bp_95ci": (round(ci_low, 2), round(ci_high, 2)),
            "net_12bp_bps": round(net_12bp, 2),
            "profitable_edge_at_8bp": bool(ci_low > 0.0),
        })

    return pd.DataFrame(records)


# ==============================================================================
# 3. Non-Overlapping Realistic Trade Simulator
# ==============================================================================

def simulate_non_overlapping_trades(
    df: pd.DataFrame,
    signals: np.ndarray,
    horizon_bars: int = 3,
    cost_bps: float = 0.0008,
) -> Dict[str, Any]:
    """
    Simulates realistic sequential execution:
    When an entry signal (+1 or -1) is taken at bar t, the trade is held for exactly
    horizon_bars (15 minutes). Any new signal arriving during bars t+1, t+2 is BLOCKED
    until the trade completes at bar t+3. Also respects same-day session boundaries.
    """
    actual_returns = df["target_return"].values
    session_dates = df["session_date"].values
    n = len(df)

    trade_indices = []
    trade_signals = []
    trade_gross_returns = []
    trade_net_returns = []

    i = 0
    while i < n:
        sig = signals[i]
        if sig != 0:
            # Active entry
            r = actual_returns[i]
            gross = sig * r
            net = gross - cost_bps

            trade_indices.append(i)
            trade_signals.append(sig)
            trade_gross_returns.append(gross)
            trade_net_returns.append(net)

            # Block entries for horizon_bars (unless session ends)
            curr_date = session_dates[i]
            skip = horizon_bars
            for step in range(1, horizon_bars + 1):
                if i + step >= n or session_dates[i + step] != curr_date:
                    skip = step
                    break
            i += skip
        else:
            i += 1

    n_trades = len(trade_net_returns)
    if n_trades == 0:
        return {
            "n_trades": 0,
            "win_rate_pct": 0.0,
            "avg_gross_bps": 0.0,
            "avg_net_bps": 0.0,
            "cumulative_net_pct": 0.0,
            "profit_factor": 0.0,
            "max_drawdown_pct": 0.0,
        }

    net_arr = np.array(trade_net_returns)
    gross_arr = np.array(trade_gross_returns)

    avg_gross_bps = float(np.mean(gross_arr) * 10000.0)
    avg_net_bps = float(np.mean(net_arr) * 10000.0)
    win_rate = float(np.mean(net_arr > 0) * 100.0)

    # Cumulative return equity curve
    equity_curve = np.cumsum(net_arr)
    cum_net_pct = float(equity_curve[-1] * 100.0)

    # Max Drawdown
    peaks = np.maximum.accumulate(equity_curve)
    drawdowns = equity_curve - peaks
    max_dd_pct = float(np.min(drawdowns) * 100.0)

    # Profit Factor
    wins = net_arr[net_arr > 0]
    losses = net_arr[net_arr < 0]
    total_win = float(np.sum(wins)) if len(wins) > 0 else 0.0
    total_loss = float(np.abs(np.sum(losses))) if len(losses) > 0 else 1e-12
    profit_factor = float(total_win / total_loss)

    return {
        "n_trades": n_trades,
        "win_rate_pct": round(win_rate, 1),
        "avg_gross_bps": round(avg_gross_bps, 2),
        "avg_net_bps": round(avg_net_bps, 2),
        "cumulative_net_pct": round(cum_net_pct, 2),
        "profit_factor": round(profit_factor, 2),
        "max_drawdown_pct": round(max_dd_pct, 2),
    }


# ==============================================================================
# 4. Synthetic Bar Sensitivity Check
# ==============================================================================

def check_filled_sensitivity(
    df: pd.DataFrame,
    signals: np.ndarray,
    cost_bps: float = 0.0008,
) -> Dict[str, Any]:
    """
    Checks if performance changes when synthetic bars (is_filled=1 or target_is_filled=1) are removed.
    """
    has_filled = (df["is_filled"] == 1) | (df.get("target_is_filled", 0) == 1)
    clean_mask = (~has_filled).values

    clean_signals = signals[clean_mask]
    clean_rets = df["target_return"].values[clean_mask]

    n_filled_bars = int(np.sum(has_filled))
    act_mask = (clean_signals != 0)

    if np.sum(act_mask) > 0:
        clean_gross = clean_signals[act_mask] * clean_rets[act_mask]
        clean_net = clean_gross - cost_bps
        clean_avg_net_bps = float(np.mean(clean_net) * 10000.0)
    else:
        clean_avg_net_bps = 0.0

    return {
        "n_synthetic_bars_dropped": n_filled_bars,
        "clean_bars_evaluated": int(np.sum(clean_mask)),
        "clean_avg_net_bps": round(clean_avg_net_bps, 2),
    }

