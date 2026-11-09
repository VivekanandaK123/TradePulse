"""
TradePulse - Step 7: Model Training & Baseline Benchmarking
Author: TradePulse Team (IIT Patna, IMPACT Lab)

Implements modular baseline models and walk-forward cross-validation:
1. Majority Class (Always Neutral)
2. Always-Up (+1)
3. Last-Return Momentum (sign of R_{t-1})
4. Basic RSI Rule (Oversold < 0.30 -> Up, Overbought > 0.70 -> Down)
5. Logistic Regression (with intra-fold RobustScaler and balanced class weights)
"""

import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.evaluate import (
    PurgedWalkForwardCV,
    compute_classification_metrics,
    compute_realized_trade_metrics,
    compute_fixed_coverage_precision,
    aggregate_fold_metrics,
)

BENCHMARK_RESULTS_PATH = PROJECT_ROOT / "artifacts" / "metadata" / "baseline_benchmark_results.json"


# ==============================================================================
# Baseline Model Wrappers (Consistent Scikit-Learn Interface)
# ==============================================================================

class MajorityBaseline:
    """Always predicts the most frequent class in the training fold."""

    def __init__(self):
        self.majority_class_ = 0

    def fit(self, X: np.ndarray, y: np.ndarray):
        values, counts = np.unique(y, return_counts=True)
        self.majority_class_ = int(values[np.argmax(counts)])
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.full(shape=(len(X),), fill_value=self.majority_class_, dtype=int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        probs = np.zeros((len(X), 3), dtype=float)
        # Class index: -1 -> 0, 0 -> 1, +1 -> 2
        col_idx = self.majority_class_ + 1
        probs[:, col_idx] = 1.0
        return probs


class AlwaysUpBaseline:
    """Always predicts Up (+1)."""

    def fit(self, X: np.ndarray, y: np.ndarray):
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.ones(shape=(len(X),), dtype=int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        probs = np.zeros((len(X), 3), dtype=float)
        probs[:, 2] = 1.0  # Class +1
        return probs


class MomentumBaseline:
    """Predicts direction of the last 1-bar return: feat_ret_1b."""

    def __init__(self, ret_col_idx: int = 0):
        self.ret_col_idx = ret_col_idx

    def fit(self, X: np.ndarray, y: np.ndarray):
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        ret = X[:, self.ret_col_idx]
        preds = np.zeros(len(X), dtype=int)
        preds[ret > 0] = 1
        preds[ret < 0] = -1
        return preds

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        preds = self.predict(X)
        probs = np.full((len(X), 3), 0.15)
        for i, p in enumerate(preds):
            probs[i, p + 1] = 0.70
        return probs


class RSIBaseline:
    """Mean-reversion rule: RSI < 0.30 -> Up (+1), RSI > 0.70 -> Down (-1), else Neutral (0)."""

    def __init__(self, rsi_col_idx: int = 14):
        self.rsi_col_idx = rsi_col_idx

    def fit(self, X: np.ndarray, y: np.ndarray):
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        rsi = X[:, self.rsi_col_idx]
        preds = np.zeros(len(X), dtype=int)
        preds[rsi < 0.30] = 1
        preds[rsi > 0.70] = -1
        return preds

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        preds = self.predict(X)
        probs = np.zeros((len(X), 3), dtype=float)
        for i, p in enumerate(preds):
            if p == 0:
                probs[i] = [0.10, 0.80, 0.10]
            else:
                probs[i, p + 1] = 0.75
                probs[i, 1] = 0.15
                probs[i, (-p) + 1] = 0.10
        return probs


class LogisticRegressionBaseline:
    """Multinomial Logistic Regression with intra-fold scaling and class weighting."""

    def __init__(self, class_weight: str = "balanced", random_state: int = 42):
        self.class_weight = class_weight
        self.random_state = random_state
        self.scaler = RobustScaler()
        self.model = LogisticRegression(
            solver="lbfgs",
            class_weight=self.class_weight,
            max_iter=1000,
            random_state=self.random_state,
        )

    def fit(self, X: np.ndarray, y: np.ndarray):
        # Scale features strictly on the training fold
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)


# ==============================================================================
# Model Evaluation Runner Across Walk-Forward CV Folds
# ==============================================================================

def evaluate_model_cv(
    df: pd.DataFrame,
    feature_cols: List[str],
    label_col: str,
    model_factory,
    n_splits: int = 5,
    purge_bars: int = 3,
) -> Dict[str, Any]:
    """
    Executes PurgedWalkForwardCV for a given model and label candidate on the training dataset.
    """
    cv = PurgedWalkForwardCV(n_splits=n_splits, purge_bars=purge_bars)
    splits = cv.split(df)

    X = df[feature_cols].values
    y = df[label_col].values.astype(int)
    actual_returns = df["target_return"].values

    fold_class_metrics = []
    fold_trade_metrics = []
    fold_fixed_precisions = []

    start_time = time.time()

    for fold_num, (train_idx, val_idx) in enumerate(splits, 1):
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        val_returns = actual_returns[val_idx]

        # Initialize and fit model
        model = model_factory()
        model.fit(X_train, y_train)

        # Predict
        preds = model.predict(X_val)
        probs = model.predict_proba(X_val)

        # Classification metrics
        c_metrics = compute_classification_metrics(y_val, preds, probs)
        fold_class_metrics.append(c_metrics)

        # Realized economic metrics (at 5, 8, and 12 bps cost)
        t_metrics = compute_realized_trade_metrics(val_returns, preds)
        fold_trade_metrics.append(t_metrics)

        # Fixed 10% coverage precision
        top_prob = np.max(probs[:, [0, 2]], axis=1)  # max probability between Down and Up
        fixed_prec = compute_fixed_coverage_precision(
            actual_returns=val_returns,
            signals=preds,
            confidences=top_prob,
            fixed_coverage_pct=0.10,
            cost_floor=0.0008,
        )
        fold_fixed_precisions.append(fixed_prec)

    elapsed = time.time() - start_time

    # Aggregate across folds
    agg_class = aggregate_fold_metrics(fold_class_metrics)
    agg_trade = aggregate_fold_metrics(fold_trade_metrics)

    summary = {
        "label_col": label_col,
        "n_folds": len(splits),
        "runtime_seconds": round(elapsed, 2),
        "macro_f1_mean": agg_class["macro_f1"]["mean"],
        "macro_f1_std": agg_class["macro_f1"]["std"],
        "balanced_acc_mean": agg_class["balanced_accuracy"]["mean"],
        "balanced_acc_std": agg_class["balanced_accuracy"]["std"],
        "log_loss_mean": agg_class["log_loss"]["mean"] if "log_loss" in agg_class else None,
        "coverage_pct_mean": agg_trade["coverage_pct"]["mean"],
        "cost_8bp_avg_trade_bps_mean": agg_trade["cost_8bp_avg_trade_bps"]["mean"],
        "cost_8bp_avg_trade_bps_std": agg_trade["cost_8bp_avg_trade_bps"]["std"],
        "cost_8bp_win_rate_mean": agg_trade["cost_8bp_win_rate_pct"]["mean"],
        "cost_8bp_profit_factor_mean": agg_trade["cost_8bp_profit_factor"]["mean"],
        "fixed_10pct_cov_prec_mean": round(float(np.mean(fold_fixed_precisions)), 2),
        "fixed_10pct_cov_prec_std": round(float(np.std(fold_fixed_precisions)), 2),
        "fold_details": {
            "classification": fold_class_metrics,
            "trade": fold_trade_metrics,
            "fixed_coverage_precision": fold_fixed_precisions,
        },
    }

    return summary


# ==============================================================================
# Comprehensive Benchmark Runner (Baselines x Label Sets)
# ==============================================================================

def run_step7_benchmark(
    train_features_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Evaluates all 5 baselines across all 4 refined candidate label sets.
    """
    from ml.features import get_feature_column_names

    csv_path = train_features_path or (PROJECT_ROOT / "data" / "processed" / "train_features.csv")
    df = pd.read_csv(csv_path)

    # Filter for valid targets
    valid_train = df[df["valid_target"]].copy().reset_index(drop=True)
    feature_cols = get_feature_column_names(valid_train)

    ret_col_idx = feature_cols.index("feat_ret_1b")
    rsi_col_idx = feature_cols.index("feat_rsi_14")

    candidate_labels = [
        "label_fixed_20bp",
        "label_quantile_50",
        "label_quantile_40",
        "label_vol_k075",
    ]

    models = [
        ("Majority (Always Neutral)", lambda: MajorityBaseline()),
        ("Always-Up", lambda: AlwaysUpBaseline()),
        ("Momentum (Sign R_t-1)", lambda: MomentumBaseline(ret_col_idx=ret_col_idx)),
        ("Basic RSI (30/70)", lambda: RSIBaseline(rsi_col_idx=rsi_col_idx)),
        ("Logistic Regression", lambda: LogisticRegressionBaseline(class_weight="balanced", random_state=42)),
    ]

    all_results = []
    table_rows = []

    print("\n[TradePulse Step 7: Baseline Benchmark Matrix]")
    print(f"  Valid training bars: {len(valid_train):,}")
    print(f"  Candidate label sets: {candidate_labels}")
    print(f"  Baselines evaluated: {[m[0] for m in models]}\n")

    for label_col in candidate_labels:
        print(f"  Evaluating candidate label set: {label_col}...")
        for model_name, model_fn in models:
            res = evaluate_model_cv(
                valid_train,
                feature_cols=feature_cols,
                label_col=label_col,
                model_factory=model_fn,
                n_splits=5,
                purge_bars=3,
            )

            res["model_name"] = model_name
            all_results.append(res)

            table_rows.append({
                "Label Candidate": label_col,
                "Model": model_name,
                "Macro F1": f"{res['macro_f1_mean']:.4f} +/- {res['macro_f1_std']:.4f}",
                "Balanced Acc": f"{res['balanced_acc_mean']:.4f} +/- {res['balanced_acc_std']:.4f}",
                "Coverage %": f"{res['coverage_pct_mean']:.1f}%",
                "8bp Net Ret/Trade": f"{res['cost_8bp_avg_trade_bps_mean']:+.2f} bps",
                "8bp Win Rate": f"{res['cost_8bp_win_rate_mean']:.1f}%",
                "10% Cov Prec": f"{res['fixed_10pct_cov_prec_mean']:.1f}%",
            })

    results_df = pd.DataFrame(table_rows)

    # Save full structured results
    BENCHMARK_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(BENCHMARK_RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print("\n=== Baseline Benchmark Results Summary ===")
    print(results_df.to_string(index=False))
    print(f"\nFull benchmark results saved to: {BENCHMARK_RESULTS_PATH}")

    return results_df


if __name__ == "__main__":
    run_step7_benchmark()

