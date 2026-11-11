"""
TradePulse - Step 9: Probability Calibration & Reliability Analysis

Implements rigorous out-of-fold calibration:
1. generate_purged_oof_probabilities: Generates out-of-fold predictions on training data using purged walk-forward CV.
2. MulticlassCalibrator: Multi-class Platt (Sigmoid) and Isotonic calibration fitted strictly on OOF training predictions.
3. evaluate_calibration_quality: Computes Brier scores, Log Loss, and reliability curves per class.
4. EnsembleCalibrator: Computes equal-weighted average of calibrated probabilities.
"""

import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss, brier_score_loss
from sklearn.calibration import calibration_curve

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.evaluate import PurgedWalkForwardCV


# ==============================================================================
# 1. Out-of-Fold Probability Generation
# ==============================================================================

def generate_purged_oof_probabilities(
    df: pd.DataFrame,
    feature_cols: List[str],
    label_col: str,
    model_factory,
    n_splits: int = 5,
    purge_bars: int = 3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[int]]:
    """
    Computes out-of-fold probability predictions on the training dataset using
    chronological PurgedWalkForwardCV with 3-bar purging.
    Returns:
        (oof_indices, oof_true_labels, oof_probs, actual_returns)
    """
    cv = PurgedWalkForwardCV(n_splits=n_splits, purge_bars=purge_bars)
    splits = cv.split(df)

    X = df[feature_cols].values
    y = df[label_col].values.astype(int)
    returns = df["target_return"].values

    oof_idx_list = []
    oof_y_list = []
    oof_probs_list = []
    oof_returns_list = []

    for fold_num, (train_idx, val_idx) in enumerate(splits, 1):
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        model = model_factory()
        model.fit(X_train, y_train)

        probs = model.predict_proba(X_val)

        oof_idx_list.extend(val_idx)
        oof_y_list.extend(y_val)
        oof_probs_list.append(probs)
        oof_returns_list.extend(returns[val_idx])

    oof_indices = np.array(oof_idx_list, dtype=int)
    oof_y = np.array(oof_y_list, dtype=int)
    oof_probs = np.vstack(oof_probs_list)
    oof_returns = np.array(oof_returns_list, dtype=float)

    return oof_indices, oof_y, oof_probs, oof_returns


# ==============================================================================
# 2. Multi-Class Calibrator (Sigmoid / Isotonic)
# ==============================================================================

class MulticlassCalibrator:
    """
    Fits one-vs-rest calibrators on uncalibrated multi-class probability vectors.
    Methods supported:
    - 'sigmoid': Platt scaling via univariate logistic regression.
    - 'isotonic': Non-parametric isotonic regression.
    """

    def __init__(self, method: str = "sigmoid"):
        if method not in ["sigmoid", "isotonic"]:
            raise ValueError(f"Method must be 'sigmoid' or 'isotonic', got {method}")
        self.method = method
        self.calibrators_: List[Any] = []
        self.classes_: np.ndarray = np.array([-1, 0, 1])

    def fit(self, probs: np.ndarray, y: np.ndarray):
        """
        Fits calibrators on uncalibrated out-of-fold probabilities and true labels.
        """
        probs = np.asarray(probs, dtype=float)
        y = np.asarray(y, dtype=int)
        self.calibrators_ = []

        for i, c in enumerate(self.classes_):
            y_binary = (y == c).astype(int)
            p_c = probs[:, i]

            if self.method == "sigmoid":
                # Logistic regression on uncalibrated probability
                cal = LogisticRegression(C=1.0, solver="lbfgs", max_iter=500)
                cal.fit(p_c.reshape(-1, 1), y_binary)
                self.calibrators_.append(cal)
            else:
                cal = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
                cal.fit(p_c, y_binary)
                self.calibrators_.append(cal)

        return self

    def predict_proba(self, probs: np.ndarray) -> np.ndarray:
        """
        Calibrates input probability array and normalizes to sum to 1.
        """
        probs = np.asarray(probs, dtype=float)
        n_samples = len(probs)
        calibrated = np.zeros_like(probs)

        for i, cal in enumerate(self.calibrators_):
            p_c = probs[:, i]
            if self.method == "sigmoid":
                calibrated[:, i] = cal.predict_proba(p_c.reshape(-1, 1))[:, 1]
            else:
                calibrated[:, i] = cal.predict(p_c)

        # Normalize across classes to preserve valid probability distribution
        row_sums = calibrated.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1e-12, row_sums)
        return calibrated / row_sums


# ==============================================================================
# 3. Calibration Evaluation Metrics & Reliability Curves
# ==============================================================================

def compute_multiclass_brier_score(y_true: np.ndarray, probs: np.ndarray, classes: List[int] = [-1, 0, 1]) -> float:
    """
    Computes mean squared error between one-hot true indicators and probability vectors.
    """
    y_true = np.asarray(y_true, dtype=int)
    probs = np.asarray(probs, dtype=float)
    n_samples = len(y_true)

    brier_sum = 0.0
    for i, c in enumerate(classes):
        y_binary = (y_true == c).astype(float)
        brier_sum += np.sum((probs[:, i] - y_binary) ** 2)

    return float(brier_sum / n_samples)


def evaluate_calibration_quality(
    y_true: np.ndarray,
    probs_uncal: np.ndarray,
    probs_cal: np.ndarray,
    classes: List[int] = [-1, 0, 1],
    n_bins: int = 10,
) -> Dict[str, Any]:
    """
    Computes log loss, Brier score, and reliability curve coordinates before and after calibration.
    """
    y_true = np.asarray(y_true, dtype=int)

    # Log Loss (clipped to avoid inf)
    eps = 1e-15
    p_uncal_clip = np.clip(probs_uncal, eps, 1 - eps)
    p_uncal_clip = p_uncal_clip / p_uncal_clip.sum(axis=1, keepdims=True)

    p_cal_clip = np.clip(probs_cal, eps, 1 - eps)
    p_cal_clip = p_cal_clip / p_cal_clip.sum(axis=1, keepdims=True)

    ll_before = float(log_loss(y_true, p_uncal_clip, labels=classes))
    ll_after = float(log_loss(y_true, p_cal_clip, labels=classes))

    brier_before = compute_multiclass_brier_score(y_true, probs_uncal, classes)
    brier_after = compute_multiclass_brier_score(y_true, probs_cal, classes)

    # Reliability curves per class
    reliability_curves = {}
    class_names = {-1: "Down (-1)", 0: "Neutral (0)", 1: "Up (+1)"}

    for i, c in enumerate(classes):
        y_bin = (y_true == c).astype(int)

        # Before calibration
        prob_true_before, prob_pred_before = calibration_curve(y_bin, probs_uncal[:, i], n_bins=n_bins, strategy="uniform")

        # After calibration
        prob_true_after, prob_pred_after = calibration_curve(y_bin, probs_cal[:, i], n_bins=n_bins, strategy="uniform")

        reliability_curves[class_names[c]] = {
            "before": {
                "prob_pred": prob_pred_before.tolist(),
                "prob_true": prob_true_before.tolist(),
            },
            "after": {
                "prob_pred": prob_pred_after.tolist(),
                "prob_true": prob_true_after.tolist(),
            },
        }

    return {
        "log_loss_before": round(ll_before, 4),
        "log_loss_after": round(ll_after, 4),
        "log_loss_reduction": round(ll_before - ll_after, 4),
        "brier_score_before": round(brier_before, 4),
        "brier_score_after": round(brier_after, 4),
        "brier_score_reduction": round(brier_before - brier_after, 4),
        "reliability_curves": reliability_curves,
    }


# ==============================================================================
# 4. Ensemble Combiner
# ==============================================================================

def create_ensemble_probabilities(probs_a: np.ndarray, probs_b: np.ndarray, weight_a: float = 0.5) -> np.ndarray:
    """
    Computes equal-weight or custom linear blend of calibrated probabilities from two models.
    """
    ens = (weight_a * probs_a) + ((1.0 - weight_a) * probs_b)
    # Ensure rows sum to 1
    return ens / ens.sum(axis=1, keepdims=True)
