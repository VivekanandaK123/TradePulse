"""
TradePulse - Step 9 & 10: Model Diagnostics & Statistical Analysis

Implements rigorous diagnostic tools for short-term financial time series:
1. TimeOfDayBaseline: Intraday 30-minute regime baseline (measures time-of-day heuristic bias).
2. Direction-Only Accuracy: Accuracy on true directional bars with binomial/bootstrap CI.
3. Dual ROC-AUC: (a) Movement vs No-movement (Neutral vs Directional), (b) Up vs Down on directional bars.
4. Time-of-Day Performance Breakdown: 30-minute intraday slots for coverage, precision, gross/net returns.
5. Paired Day-Level Block Bootstrap: Resamples entire trading days (with replacement) to compute 95% CIs for differences between models.
"""

import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ==============================================================================
# 1. Time-of-Day Baseline Model
# ==============================================================================

class TimeOfDayBaseline:
    """
    Predicts class probabilities based strictly on 30-minute intraday time slots.
    Evaluates how much predictive capacity is merely learning 'volatile at open/close, quiet at midday'.
    """

    def __init__(self, slot_minutes: int = 30):
        self.slot_minutes = slot_minutes
        self.slot_probs_: Dict[int, np.ndarray] = {}
        self.global_probs_: np.ndarray = np.array([0.25, 0.50, 0.25])

    def _get_slot(self, timestamps_ny: Union[pd.Series, np.ndarray, pd.DatetimeIndex]) -> np.ndarray:
        s = pd.Series(pd.to_datetime(timestamps_ny, utc=True)).dt.tz_convert("America/New_York")
        minutes_since_open = (s.dt.hour - 9) * 60 + (s.dt.minute - 30)
        return (minutes_since_open // self.slot_minutes).clip(lower=0, upper=12).values

    def fit(self, timestamps_ny: pd.Series, y: np.ndarray):
        slots = self._get_slot(timestamps_ny)
        y = np.asarray(y, dtype=int)
        
        # Global prior
        classes = [-1, 0, 1]
        for c in classes:
            self.global_probs_[c + 1] = np.mean(y == c)

        for s in range(13):
            mask = (slots == s)
            if np.sum(mask) > 0:
                y_s = y[mask]
                probs = np.array([np.mean(y_s == c) for c in classes])
                self.slot_probs_[s] = probs
            else:
                self.slot_probs_[s] = self.global_probs_.copy()
        return self

    def predict_proba(self, timestamps_ny: Union[pd.Series, np.ndarray, pd.DatetimeIndex]) -> np.ndarray:
        slots = self._get_slot(timestamps_ny)
        n = len(slots)
        probs = np.zeros((n, 3), dtype=float)
        for i, s in enumerate(slots):
            probs[i] = self.slot_probs_.get(s, self.global_probs_)
        return probs

    def predict(self, timestamps_ny: pd.Series) -> np.ndarray:
        probs = self.predict_proba(timestamps_ny)
        classes = np.array([-1, 0, 1])
        return classes[np.argmax(probs, axis=1)]


# ==============================================================================
# 2. Direction-Only Accuracy
# ==============================================================================

def compute_direction_only_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Evaluates prediction accuracy strictly among bars that were TRULY directional (y_true in {-1, +1}).
    Checks whether the model predicts movement direction, or merely predicts volatility.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    mask = (y_true != 0)
    n_dir = int(np.sum(mask))

    if n_dir == 0:
        return {"n_directional_bars": 0, "direction_accuracy": 0.0, "ci_lower": 0.0, "ci_upper": 0.0}

    # On truly directional bars, how often did predicted sign match true sign?
    # Neutral predictions count as 0 match
    matches = (np.sign(y_pred[mask]) == np.sign(y_true[mask])).astype(float)
    acc = float(np.mean(matches))

    # Clopper-Pearson exact binomial confidence interval
    k = int(np.sum(matches))
    ci_lower, ci_upper = stats.beta.interval(1 - alpha, k + 1, n_dir - k + 1)
    ci_lower = max(0.0, float(ci_lower))
    ci_upper = min(1.0, float(ci_upper))

    return {
        "n_directional_bars": n_dir,
        "direction_accuracy": round(acc, 4),
        "ci_lower": round(ci_lower, 4),
        "ci_upper": round(ci_upper, 4),
        "is_distinguishable_from_coinflip": bool(ci_lower > 0.50 or ci_upper < 0.50),
    }


# ==============================================================================
# 3. Dual ROC-AUC (Move vs No-Move, and Up vs Down)
# ==============================================================================

def compute_dual_roc_auc(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> Dict[str, Optional[float]]:
    """
    Computes two distinct ROC-AUC scores:
    1. Move vs. No-Move: Neutral (0) vs Directional (-1 or +1 combined).
       Score is P(Directional) = P(-1) + P(+1).
    2. Up vs. Down: Strictly among directional bars (-1 vs +1).
       Score is P(+1) / (P(-1) + P(+1)).
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    # 1. Move vs No-Move
    y_move = (y_true != 0).astype(int)
    p_move = y_prob[:, 0] + y_prob[:, 2]

    try:
        auc_move = float(roc_auc_score(y_move, p_move))
    except Exception:
        auc_move = None

    # 2. Up vs Down (on directional bars only)
    dir_mask = (y_true != 0)
    if np.sum(y_true == 1) > 0 and np.sum(y_true == -1) > 0:
        y_dir = (y_true[dir_mask] == 1).astype(int)  # 1: Up, 0: Down
        denom = y_prob[dir_mask, 0] + y_prob[dir_mask, 2]
        denom = np.where(denom == 0, 1e-12, denom)
        p_up_rel = y_prob[dir_mask, 2] / denom
        try:
            auc_up_down = float(roc_auc_score(y_dir, p_up_rel))
        except Exception:
            auc_up_down = None
    else:
        auc_up_down = None

    return {
        "auc_move_vs_nomove": round(auc_move, 4) if auc_move is not None else None,
        "auc_up_vs_down": round(auc_up_down, 4) if auc_up_down is not None else None,
    }


# ==============================================================================
# 4. Performance Breakdown by Time of Day (30-Minute Slots)
# ==============================================================================

def compute_performance_by_tod(
    timestamps_ny: pd.Series,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    actual_returns: np.ndarray,
    cost_bps: float = 0.0008,
) -> pd.DataFrame:
    """
    Evaluates model performance across 30-minute intraday slots (09:30-10:00 to 15:30-16:00).
    Reveals whether any edge exists outside the opening hour.
    """
    s = pd.Series(pd.to_datetime(timestamps_ny, utc=True)).dt.tz_convert("America/New_York")
    minutes = (s.dt.hour - 9) * 60 + (s.dt.minute - 30)
    slot_idx = (minutes // 30).clip(lower=0, upper=12).values

    slot_labels = [
        "09:30-10:00", "10:00-10:30", "10:30-11:00", "11:00-11:30",
        "11:30-12:00", "12:00-12:30", "12:30-13:00", "13:00-13:30",
        "13:30-14:00", "14:00-14:30", "14:30-15:00", "15:00-15:30",
        "15:30-16:00"
    ]

    records = []
    for s in range(13):
        mask = (slot_idx == s)
        n_bars = int(np.sum(mask))
        if n_bars == 0:
            continue

        s_y_true = y_true[mask]
        s_y_pred = y_pred[mask]
        s_rets = actual_returns[mask]

        active_mask = (s_y_pred != 0)
        n_active = int(np.sum(active_mask))
        cov_pct = (n_active / n_bars) * 100.0

        if n_active > 0:
            act_preds = s_y_pred[active_mask]
            act_trues = s_y_true[active_mask]
            act_rets = s_rets[active_mask]

            gross_rets = act_preds * act_rets
            net_rets = gross_rets - cost_bps
            prec = np.mean(act_preds == act_trues) * 100.0
            gross_bps = np.mean(gross_rets) * 10000.0
            net_bps = np.mean(net_rets) * 10000.0
            win_rate = np.mean(net_rets > 0) * 100.0
        else:
            prec = 0.0
            gross_bps = 0.0
            net_bps = 0.0
            win_rate = 0.0

        records.append({
            "slot_idx": s,
            "slot_time": slot_labels[s],
            "total_bars": n_bars,
            "active_trades": n_active,
            "coverage_pct": round(cov_pct, 1),
            "precision_pct": round(prec, 1),
            "gross_return_bps": round(gross_bps, 2),
            "net_return_8bp_bps": round(net_bps, 2),
            "win_rate_pct": round(win_rate, 1),
        })

    return pd.DataFrame(records)


# ==============================================================================
# 5. Paired Day-Level Block Bootstrap
# ==============================================================================

def paired_day_block_bootstrap(
    session_dates: pd.Series,
    y_true: np.ndarray,
    actual_returns: np.ndarray,
    preds_a: np.ndarray,
    preds_b: np.ndarray,
    cost_bps: float = 0.0008,
    n_bootstraps: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Performs paired block bootstrap resampling by entire TRADING DAYS (session_date),
    preserving all intraday cross-correlations and volatility clustering.
    Evaluates whether the difference between Model A and Model B in:
    (a) Gross return per signal, and (b) Macro-F1
    is distinguishable from zero at alpha = 0.05.
    """
    rng = np.random.default_rng(seed)
    unique_dates = np.unique(session_dates)
    n_days = len(unique_dates)

    df_data = pd.DataFrame({
        "date": session_dates.values,
        "y_true": y_true,
        "ret": actual_returns,
        "pred_a": preds_a,
        "pred_b": preds_b,
    })

    # Group records by date for fast resampling
    day_groups = [group for _, group in df_data.groupby("date")]

    diff_gross_returns = []
    diff_macro_f1s = []

    for _ in range(n_bootstraps):
        sampled_indices = rng.choice(n_days, size=n_days, replace=True)
        sample_df = pd.concat([day_groups[i] for i in sampled_indices], ignore_index=True)

        y_t = sample_df["y_true"].values
        ret = sample_df["ret"].values
        p_a = sample_df["pred_a"].values
        p_b = sample_df["pred_b"].values

        # 1. Macro-F1 difference
        f1_a = f1_score(y_t, p_a, average="macro", zero_division=0)
        f1_b = f1_score(y_t, p_b, average="macro", zero_division=0)
        diff_macro_f1s.append(f1_a - f1_b)

        # 2. Gross return per signal difference
        mask_a = (p_a != 0)
        mask_b = (p_b != 0)

        gross_a = np.mean(p_a[mask_a] * ret[mask_a]) if np.sum(mask_a) > 0 else 0.0
        gross_b = np.mean(p_b[mask_b] * ret[mask_b]) if np.sum(mask_b) > 0 else 0.0
        diff_gross_returns.append((gross_a - gross_b) * 10000.0)

    diff_f1 = np.array(diff_macro_f1s)
    diff_ret = np.array(diff_gross_returns)

    # 95% Percentile Confidence Intervals
    f1_ci = (float(np.percentile(diff_f1, 2.5)), float(np.percentile(diff_f1, 97.5)))
    ret_ci = (float(np.percentile(diff_ret, 2.5)), float(np.percentile(diff_ret, 97.5)))

    # Distinguishable from zero?
    f1_significant = not (f1_ci[0] <= 0.0 <= f1_ci[1])
    ret_significant = not (ret_ci[0] <= 0.0 <= ret_ci[1])

    return {
        "n_bootstraps": n_bootstraps,
        "n_days_sampled": n_days,
        "diff_macro_f1_mean": round(float(np.mean(diff_f1)), 4),
        "diff_macro_f1_95ci": (round(f1_ci[0], 4), round(f1_ci[1], 4)),
        "macro_f1_distinguishable_from_zero": f1_significant,
        "diff_gross_return_bps_mean": round(float(np.mean(diff_ret)), 2),
        "diff_gross_return_bps_95ci": (round(ret_ci[0], 2), round(ret_ci[1], 2)),
        "gross_return_distinguishable_from_zero": ret_significant,
    }
