"""
TradePulse - Step 7: Evaluation Framework & Purged Walk-Forward Cross-Validation

Implements rigorous evaluation metrics and validation splitting:
1. PurgedWalkForwardCV: Expanding-window time series splitter with mandatory 3-bar purging.
2. Classification Metrics: Balanced accuracy, Macro-F1, per-class Precision/Recall/F1, Log Loss, Brier score.
3. Cross-Label Economic Metrics: Cost-adjusted realized returns (at 5, 8, 12 bps) and fixed-coverage precision.
"""

from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    log_loss,
    brier_score_loss,
)


# ==============================================================================
# Purged Walk-Forward Time-Series Cross-Validation Splitter
# ==============================================================================

class PurgedWalkForwardCV:
    """
    Chronological expanding-window cross-validation with gap purging.
    Ensures that for a 3-bar forward target (R_{t, t+3}), at least 3 bars
    between the end of the train fold and the start of the validation fold
    are completely dropped (purged) to eliminate lookahead target overlap.
    """

    def __init__(
        self,
        n_splits: int = 5,
        purge_bars: int = 3,
        min_train_fraction: float = 0.40,
    ):
        self.n_splits = n_splits
        self.purge_bars = purge_bars
        self.min_train_fraction = min_train_fraction

    def split(self, X: pd.DataFrame) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Yields (train_indices, val_indices) pairs.
        """
        n_samples = len(X)
        val_size = int(n_samples * ((1.0 - self.min_train_fraction) / self.n_splits))
        min_train = n_samples - (self.n_splits * val_size)

        splits = []
        for i in range(self.n_splits):
            train_end = min_train + (i * val_size)
            val_start = train_end + self.purge_bars
            val_end = min(val_start + val_size, n_samples)

            if val_start >= n_samples:
                break

            train_idx = np.arange(0, train_end)
            val_idx = np.arange(val_start, val_end)

            splits.append((train_idx, val_idx))

        return splits


# ==============================================================================
# Classification Performance Metrics
# ==============================================================================

def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    classes: List[int] = [-1, 0, 1],
) -> Dict[str, Any]:
    """
    Computes balanced classification metrics (Macro-F1, Balanced Accuracy,
    Per-class precision and recall, Confusion Matrix, and Log Loss).
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    acc = float(accuracy_score(y_true, y_pred))
    b_acc = float(balanced_accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    # Per-class metrics: -1 (Down), 0 (Neutral), 1 (Up)
    prec_per_class = precision_score(y_true, y_pred, labels=classes, average=None, zero_division=0)
    rec_per_class = recall_score(y_true, y_pred, labels=classes, average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, labels=classes, average=None, zero_division=0)

    cm = confusion_matrix(y_true, y_pred, labels=classes).tolist()

    metrics: Dict[str, Any] = {
        "accuracy": round(acc, 4),
        "balanced_accuracy": round(b_acc, 4),
        "macro_f1": round(macro_f1, 4),
        "down_precision": round(float(prec_per_class[0]), 4),
        "down_recall": round(float(rec_per_class[0]), 4),
        "down_f1": round(float(f1_per_class[0]), 4),
        "neutral_precision": round(float(prec_per_class[1]), 4),
        "neutral_recall": round(float(rec_per_class[1]), 4),
        "neutral_f1": round(float(f1_per_class[1]), 4),
        "up_precision": round(float(prec_per_class[2]), 4),
        "up_recall": round(float(rec_per_class[2]), 4),
        "up_f1": round(float(f1_per_class[2]), 4),
        "confusion_matrix": cm,
    }

    # Probabilistic metrics
    if y_prob is not None:
        y_prob = np.asarray(y_prob, dtype=float)
        # Clip probabilities to avoid infinite log_loss
        eps = 1e-15
        y_prob = np.clip(y_prob, eps, 1.0 - eps)
        # Normalize rows to sum to 1
        y_prob = y_prob / y_prob.sum(axis=1, keepdims=True)

        try:
            ll = float(log_loss(y_true, y_prob, labels=classes))
            metrics["log_loss"] = round(ll, 4)
        except Exception:
            metrics["log_loss"] = None

        # Multiclass Brier score: mean squared error across one-hot indicator vectors
        try:
            y_one_hot = np.zeros_like(y_prob)
            for idx, c in enumerate(classes):
                y_one_hot[:, idx] = (y_true == c).astype(float)
            brier = float(np.mean(np.sum((y_prob - y_one_hot) ** 2, axis=1)))
            metrics["brier_score"] = round(brier, 4)
        except Exception:
            metrics["brier_score"] = None
    else:
        metrics["log_loss"] = None
        metrics["brier_score"] = None

    return metrics


# ==============================================================================
# Cross-Label Economic & Price Realization Metrics
# ==============================================================================

def compute_realized_trade_metrics(
    actual_returns: pd.Series,
    signals: pd.Series,
    cost_levels: List[float] = [0.0005, 0.0008, 0.0012],
) -> Dict[str, Any]:
    """
    Computes label-independent trading performance based strictly on actual
    future 15-minute price changes:
    - Up (+1): Long position -> +R_{t, t+3} - cost
    - Down (-1): Short position -> -R_{t, t+3} - cost
    - Neutral (0): No position -> 0.0 return
    """
    actual_r = np.asarray(actual_returns, dtype=float)
    sigs = np.asarray(signals, dtype=float)

    active_mask = (sigs == 1.0) | (sigs == -1.0)
    total_bars = len(sigs)
    n_signals = int(active_mask.sum())
    coverage_pct = (n_signals / total_bars * 100.0) if total_bars > 0 else 0.0

    trade_returns_raw = np.where(sigs == 1.0, actual_r, np.where(sigs == -1.0, -actual_r, 0.0))

    metrics: Dict[str, Any] = {
        "total_bars": total_bars,
        "n_active_signals": n_signals,
        "coverage_pct": round(coverage_pct, 2),
    }

    for cost in cost_levels:
        cost_bps = int(round(cost * 10000))
        # Deduct cost only when a trade is taken
        trade_returns_net = np.where(active_mask, trade_returns_raw - cost, 0.0)

        if n_signals > 0:
            net_active_trades = trade_returns_net[active_mask]
            win_rate = float((net_active_trades > 0).mean() * 100.0)
            avg_return_per_trade_bps = float(net_active_trades.mean() * 10000)
            cum_return_pct = float(np.sum(net_active_trades) * 100.0)

            # Profit factor
            gains = net_active_trades[net_active_trades > 0].sum()
            losses = np.abs(net_active_trades[net_active_trades < 0].sum())
            pf = float(gains / losses) if losses > 0 else (99.0 if gains > 0 else 0.0)
        else:
            win_rate = 0.0
            avg_return_per_trade_bps = 0.0
            cum_return_pct = 0.0
            pf = 0.0

        metrics[f"cost_{cost_bps}bp_avg_trade_bps"] = round(avg_return_per_trade_bps, 2)
        metrics[f"cost_{cost_bps}bp_win_rate_pct"] = round(win_rate, 2)
        metrics[f"cost_{cost_bps}bp_cum_return_pct"] = round(cum_return_pct, 2)
        metrics[f"cost_{cost_bps}bp_profit_factor"] = round(pf, 3)

    return metrics


def compute_fixed_coverage_precision(
    actual_returns: pd.Series,
    signals: pd.Series,
    confidences: Optional[pd.Series] = None,
    fixed_coverage_pct: float = 0.10,
    cost_floor: float = 0.0008,
) -> float:
    """
    Computes Up/Down precision when evaluated on a fixed fraction of bars (e.g. top 10%).
    A signal is counted as correct if the realized move in the signaled direction
    exceeds the assumed round-trip cost floor (+0.08%).
    """
    actual_r = np.asarray(actual_returns, dtype=float)
    sigs = np.asarray(signals, dtype=float)

    if confidences is not None:
        confs = np.asarray(confidences, dtype=float)
    else:
        confs = np.abs(sigs)

    # Consider only bars where an Up or Down signal was emitted
    active_idx = np.where((sigs == 1.0) | (sigs == -1.0))[0]
    if len(active_idx) == 0:
        return 0.0

    # Determine target number of bars for fixed coverage
    target_count = max(1, int(len(sigs) * fixed_coverage_pct))

    # Sort active signals by confidence descending
    active_confs = confs[active_idx]
    sorted_order = np.argsort(-active_confs)
    selected_idx = active_idx[sorted_order[:target_count]]

    # Check whether actual price moved in direction by more than cost floor
    sel_sigs = sigs[selected_idx]
    sel_returns = actual_r[selected_idx]

    correct_direction_with_edge = (
        ((sel_sigs == 1.0) & (sel_returns > cost_floor)) |
        ((sel_sigs == -1.0) & (sel_returns < -cost_floor))
    )

    precision = float(correct_direction_with_edge.mean() * 100.0)
    return round(precision, 2)


# ==============================================================================
# Aggregate Fold Metrics Utility
# ==============================================================================

def aggregate_fold_metrics(fold_metrics: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """
    Computes mean and spread (standard deviation) across cross-validation folds.
    """
    summary: Dict[str, Dict[str, float]] = {}
    keys = [k for k in fold_metrics[0].keys() if k != "confusion_matrix"]

    for k in keys:
        values = [fm[k] for fm in fold_metrics if fm.get(k) is not None]
        if values:
            summary[k] = {
                "mean": round(float(np.mean(values)), 4),
                "std": round(float(np.std(values)), 4),
                "min": round(float(np.min(values)), 4),
                "max": round(float(np.max(values)), 4),
            }

    return summary

