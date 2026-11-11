"""
TradePulse - Detailed Model Comparison on Selected Target (label_fixed_15bp)
Evaluates Baselines, LR, Tuned RF, Tuned XGBoost, and Ensemble on both
Validation and 5-Fold Purged Walk-Forward Cross-Validation.
"""

import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score, log_loss

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.features import get_feature_column_names
from ml.evaluate import PurgedWalkForwardCV, compute_classification_metrics
from ml.train import (
    LogisticRegressionBaseline,
    RandomForestModel,
    XGBoostModel,
    MomentumBaseline,
    AlwaysUpBaseline,
    MajorityBaseline,
    RSIBaseline,
)
from ml.diagnostics import TimeOfDayBaseline
from ml.calibration import MulticlassCalibrator, create_ensemble_probabilities
from ml.label_expansion_experiments import (
    compute_directional_conviction,
    apply_fixed_coverage_signals,
    compute_binary_sign_accuracy,
    compute_up_down_roc_auc,
)

METADATA_DIR = PROJECT_ROOT / "artifacts" / "metadata"
TARGET_COL = "label_fixed_15bp"


def compute_economic_metrics(signals: np.ndarray, returns: np.ndarray) -> Dict[str, Any]:
    active = (signals != 0)
    n_sig = int(np.sum(active))
    cov_pct = float(n_sig / len(signals) * 100.0)

    if n_sig == 0:
        return {
            "coverage_pct": 0.0,
            "gross_return_bps": 0.0,
            "net_5bp_bps": 0.0,
            "net_8bp_bps": 0.0,
            "net_12bp_bps": 0.0,
            "win_rate_8bp_pct": 0.0,
        }

    act_sig = signals[active]
    act_ret = returns[active]
    gross = act_sig * act_ret
    gross_bps = float(np.mean(gross) * 10000.0)
    net_5bp = float(np.mean(gross - 0.0005) * 10000.0)
    net_8bp = float(np.mean(gross - 0.0008) * 10000.0)
    net_12bp = float(np.mean(gross - 0.0012) * 10000.0)
    win_rate_8bp = float(np.mean((gross - 0.0008) > 0) * 100.0)

    return {
        "coverage_pct": round(cov_pct, 2),
        "gross_return_bps": round(gross_bps, 2),
        "net_5bp_bps": round(net_5bp, 2),
        "net_8bp_bps": round(net_8bp, 2),
        "net_12bp_bps": round(net_12bp, 2),
        "win_rate_8bp_pct": round(win_rate_8bp, 1),
    }


def compute_top10_metrics(probs: Optional[np.ndarray], returns: np.ndarray, y_true: np.ndarray) -> Dict[str, Any]:
    if probs is None:
        return {"top10_prec": "N/A", "top10_gross_bps": "N/A", "top10_net_8bp_bps": "N/A"}

    # Top 10% conviction selection
    signals, cutoff = apply_fixed_coverage_signals(probs, top_fraction=0.10)
    active = (signals != 0)
    if np.sum(active) == 0:
        return {"top10_prec": "N/A", "top10_gross_bps": "N/A", "top10_net_8bp_bps": "N/A"}

    # Precision definition from Step 8 report: signal matches actual price move >= 8 bps
    act_sig = signals[active]
    act_ret = returns[active]
    matches = ((act_sig == 1) & (act_ret >= 0.0008)) | ((act_sig == -1) & (act_ret <= -0.0008))
    prec = float(np.mean(matches) * 100.0)

    gross = act_sig * act_ret
    gross_bps = float(np.mean(gross) * 10000.0)
    net_8bp = float(np.mean(gross - 0.0008) * 10000.0)

    return {
        "top10_prec": f"{prec:.1f}%",
        "top10_gross_bps": f"{gross_bps:+.2f} bps",
        "top10_net_8bp_bps": f"{net_8bp:+.2f} bps",
    }


def evaluate_model_pipeline():
    print(f"Loading data for target: {TARGET_COL}...")
    train_df = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "train_features.csv")
    val_df = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "val_features.csv")

    valid_train = train_df[train_df["valid_target"]].copy().reset_index(drop=True)
    valid_val = val_df[val_df["valid_target"]].copy().reset_index(drop=True)
    feature_cols = get_feature_column_names(valid_train)

    X_train = valid_train[feature_cols].values
    y_train = valid_train[TARGET_COL].values.astype(int)
    r_train = valid_train["target_return"].values

    X_val = valid_val[feature_cols].values
    y_val = valid_val[TARGET_COL].values.astype(int)
    r_val = valid_val["target_return"].values
    t_val = valid_val["timestamp_ny"]

    print(f"Train valid bars: {len(valid_train):,} | Val valid bars: {len(valid_val):,}")

    # Model Factories
    rf_factory = lambda: RandomForestModel(n_estimators=150, max_depth=3, min_samples_leaf=50, max_features="sqrt", class_weight="balanced", random_state=42)
    xgb_factory = lambda: XGBoostModel(n_estimators=100, max_depth=4, learning_rate=0.10, subsample=0.8, colsample_bytree=0.8, min_child_weight=1.0, random_state=42)
    lr_factory = lambda: LogisticRegressionBaseline(C=1.0, class_weight="balanced", random_state=42)

    # 1. 5-Fold Purged Walk-Forward Cross-Validation on Training Set
    cv = PurgedWalkForwardCV(n_splits=5, purge_bars=3)
    splits = cv.split(valid_train)

    models_list = [
        "Majority (Always Neutral)",
        "Always-Up (+1)",
        "Momentum (Sign R_t-1)",
        "Basic RSI (30/70)",
        "Time-of-Day Baseline",
        "Logistic Regression",
        "Random Forest (Tuned)",
        "XGBoost (Tuned)",
        "Ensemble (50% RF + 50% LR)",
    ]

    cv_results = {m: [] for m in models_list}

    print("\nRunning 5-Fold Purged Walk-Forward CV...")
    rsi_col_idx = feature_cols.index("feat_rsi_14")

    for fold, (tr_idx, va_idx) in enumerate(splits, 1):
        X_tr, y_tr = X_train[tr_idx], y_train[tr_idx]
        X_va, y_va = X_train[va_idx], y_train[va_idx]
        r_va = r_train[va_idx]
        t_tr = valid_train["timestamp_ny"].iloc[tr_idx]
        t_va = valid_train["timestamp_ny"].iloc[va_idx]

        # Fit models
        m_lr = lr_factory().fit(X_tr, y_tr)
        m_rf = rf_factory().fit(X_tr, y_tr)
        m_xgb = xgb_factory().fit(X_tr, y_tr)

        m_maj = MajorityBaseline().fit(X_tr, y_tr)
        m_up = AlwaysUpBaseline().fit(X_tr, y_tr)
        m_mom = MomentumBaseline().fit(X_tr, y_tr)
        m_rsi = RSIBaseline(rsi_col_idx=rsi_col_idx).fit(X_tr, y_tr)
        m_tod = TimeOfDayBaseline().fit(t_tr, y_tr)

        # Predict
        p_lr = m_lr.predict_proba(X_va)
        p_rf = m_rf.predict_proba(X_va)
        p_xgb = m_xgb.predict_proba(X_va)
        p_ens = create_ensemble_probabilities(p_rf, p_lr)
        p_tod = m_tod.predict_proba(t_va)

        preds = {
            "Majority (Always Neutral)": (m_maj.predict(X_va), None),
            "Always-Up (+1)": (m_up.predict(X_va), None),
            "Momentum (Sign R_t-1)": (m_mom.predict(X_va), None),
            "Basic RSI (30/70)": (m_rsi.predict(X_va), None),
            "Time-of-Day Baseline": (m_tod.predict(t_va), p_tod),
            "Logistic Regression": (m_lr.predict(X_va), p_lr),
            "Random Forest (Tuned)": (m_rf.predict(X_va), p_rf),
            "XGBoost (Tuned)": (m_xgb.predict(X_va), p_xgb),
            "Ensemble (50% RF + 50% LR)": (np.array([-1, 0, 1])[np.argmax(p_ens, axis=1)], p_ens),
        }

        for m_name, (pred, prob) in preds.items():
            clf_m = compute_classification_metrics(y_va, pred, prob)
            econ_m = compute_economic_metrics(pred, r_va)
            top10_m = compute_top10_metrics(prob, r_va, y_va)
            sign_m = compute_binary_sign_accuracy(y_va, prob) if prob is not None else {"sign_acc": 50.0}
            updown_auc = compute_up_down_roc_auc(y_va, prob) if prob is not None else None

            cv_results[m_name].append({
                "accuracy": clf_m["accuracy"],
                "balanced_accuracy": clf_m["balanced_accuracy"],
                "macro_f1": clf_m["macro_f1"],
                "coverage_pct": econ_m["coverage_pct"],
                "gross_bps": econ_m["gross_return_bps"],
                "net_8bp_bps": econ_m["net_8bp_bps"],
                "win_rate_8bp": econ_m["win_rate_8bp_pct"],
                "top10_prec": top10_m["top10_prec"],
                "sign_acc": sign_m["sign_acc"],
                "updown_auc": updown_auc,
            })

    # 2. Validation Set Evaluation (Fit on Full Training Set)
    print("\nFitting full models on entire training set...")
    full_lr = lr_factory().fit(X_train, y_train)
    full_rf = rf_factory().fit(X_train, y_train)
    full_xgb = xgb_factory().fit(X_train, y_train)

    full_maj = MajorityBaseline().fit(X_train, y_train)
    full_up = AlwaysUpBaseline().fit(X_train, y_train)
    full_mom = MomentumBaseline().fit(X_train, y_train)
    full_rsi = RSIBaseline(rsi_col_idx=rsi_col_idx).fit(X_train, y_train)
    full_tod = TimeOfDayBaseline().fit(valid_train["timestamp_ny"], y_train)

    val_p_lr = full_lr.predict_proba(X_val)
    val_p_rf = full_rf.predict_proba(X_val)
    val_p_xgb = full_xgb.predict_proba(X_val)
    val_p_ens = create_ensemble_probabilities(val_p_rf, val_p_lr)
    val_p_tod = full_tod.predict_proba(t_val)

    # Calibrators fit on OOF (from label expansion experiments)
    cal_rf = MulticlassCalibrator(method="sigmoid").fit(val_p_rf, y_val)  # evaluation will compute before vs after

    val_preds_dict = {
        "Majority (Always Neutral)": (full_maj.predict(X_val), None),
        "Always-Up (+1)": (full_up.predict(X_val), None),
        "Momentum (Sign R_t-1)": (full_mom.predict(X_val), None),
        "Basic RSI (30/70)": (full_rsi.predict(X_val), None),
        "Time-of-Day Baseline": (full_tod.predict(t_val), val_p_tod),
        "Logistic Regression": (full_lr.predict(X_val), val_p_lr),
        "Random Forest (Tuned)": (full_rf.predict(X_val), val_p_rf),
        "XGBoost (Tuned)": (full_xgb.predict(X_val), val_p_xgb),
        "Ensemble (50% RF + 50% LR)": (np.array([-1, 0, 1])[np.argmax(val_p_ens, axis=1)], val_p_ens),
    }

    val_results = {}
    for m_name, (pred, prob) in val_preds_dict.items():
        clf_m = compute_classification_metrics(y_val, pred, prob)
        econ_m = compute_economic_metrics(pred, r_val)
        top10_m = compute_top10_metrics(prob, r_val, y_val)
        sign_m = compute_binary_sign_accuracy(y_val, prob) if prob is not None else {"sign_acc": 50.0, "ci_low": 50.0, "ci_high": 50.0}
        updown_auc = compute_up_down_roc_auc(y_val, prob) if prob is not None else None

        val_results[m_name] = {
            "classification": clf_m,
            "economic": econ_m,
            "top10": top10_m,
            "sign_accuracy": sign_m,
            "updown_auc": updown_auc,
        }

    # Aggregate CV results (mean ± std)
    cv_summary = {}
    for m_name, records in cv_results.items():
        cv_summary[m_name] = {
            "accuracy": f"{np.mean([r['accuracy'] for r in records])*100:.1f}% ± {np.std([r['accuracy'] for r in records])*100:.1f}%",
            "balanced_accuracy": f"{np.mean([r['balanced_accuracy'] for r in records])*100:.1f}% ± {np.std([r['balanced_accuracy'] for r in records])*100:.1f}%",
            "macro_f1": f"{np.mean([r['macro_f1'] for r in records]):.4f} ± {np.std([r['macro_f1'] for r in records]):.4f}",
            "coverage_pct": f"{np.mean([r['coverage_pct'] for r in records]):.1f}%",
            "gross_bps": f"{np.mean([r['gross_bps'] for r in records]):+.2f} bps",
            "net_8bp_bps": f"{np.mean([r['net_8bp_bps'] for r in records]):+.2f} bps",
            "win_rate_8bp": f"{np.mean([r['win_rate_8bp'] for r in records]):.1f}%",
            "top10_prec": records[0]["top10_prec"] if records[0]["top10_prec"] == "N/A" else f"{np.mean([float(r['top10_prec'].replace('%','')) for r in records if r['top10_prec'] != 'N/A']):.1f}%",
            "sign_acc": f"{np.mean([r['sign_acc'] for r in records]):.1f}%",
            "updown_auc": f"{np.mean([r['updown_auc'] for r in records if r['updown_auc'] is not None]):.4f}" if any(r['updown_auc'] is not None for r in records) else "N/A",
        }

    output_data = {
        "target_label": TARGET_COL,
        "n_train": len(valid_train),
        "n_val": len(valid_val),
        "validation_results": val_results,
        "cv_summary": cv_summary,
    }

    out_file = METADATA_DIR / "model_comparison_label_fixed_15bp_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nSaved comparison results to: {out_file}")
    return output_data


if __name__ == "__main__":
    evaluate_model_pipeline()
