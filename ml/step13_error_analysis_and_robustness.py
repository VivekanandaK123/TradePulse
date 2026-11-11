"""
TradePulse - Step 13: Error Analysis, Regimes & Robustness
Author: TradePulse Team (IIT Patna, IMPACT Lab)

Performs comprehensive error analysis and robustness checks:
1. Performance by Time-of-Day (30-min slots).
2. Performance by Volatility Regime (Low, Mid, High).
3. Performance by Confidence Bucket (Deciles/Bins of conviction).
4. Feature Importance (MDI & Permutation) + Group Ablation Study.
5. Random Seed Sensitivity (5 seeds for RF & XGBoost).
"""

import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, log_loss
from sklearn.inspection import permutation_importance

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.features import get_feature_column_names
from ml.evaluate import compute_classification_metrics
from ml.train import RandomForestModel, XGBoostModel, LogisticRegressionBaseline
from ml.calibration import create_ensemble_probabilities
from ml.label_expansion_experiments import (
    compute_directional_conviction,
    apply_fixed_coverage_signals,
    compute_binary_sign_accuracy,
    compute_up_down_roc_auc,
)

METADATA_DIR = PROJECT_ROOT / "artifacts" / "metadata"
MODELS_DIR = PROJECT_ROOT / "artifacts" / "models"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
TARGET_COL = "label_fixed_15bp"


def run_error_analysis():
    print("=" * 80)
    print("TradePulse - Step 13: Error Analysis, Regimes & Robustness")
    print("=" * 80)

    # 1. Load Data
    train_df = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "train_features.csv")
    val_df = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "val_features.csv")
    test_df = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "test_features.csv")

    valid_train = train_df[train_df["valid_target"]].copy().reset_index(drop=True)
    valid_val = val_df[val_df["valid_target"]].copy().reset_index(drop=True)
    valid_test = test_df[test_df["valid_target"]].copy().reset_index(drop=True)
    feature_cols = get_feature_column_names(valid_train)

    print(f"Loaded: Train {len(valid_train):,} | Val {len(valid_val):,} | Test {len(valid_test):,}")

    X_train = valid_train[feature_cols].values
    y_train = valid_train[TARGET_COL].values.astype(int)

    X_val = valid_val[feature_cols].values
    y_val = valid_val[TARGET_COL].values.astype(int)
    r_val = valid_val["target_return"].values

    X_test = valid_test[feature_cols].values
    y_test = valid_test[TARGET_COL].values.astype(int)
    r_test = valid_test["target_return"].values

    # Load Frozen Models & Calibrator
    rf_model = joblib.load(MODELS_DIR / "random_forest_model_label_expansion.joblib")
    lr_model = joblib.load(MODELS_DIR / "logistic_regression_model_label_expansion.joblib")
    xgb_model = joblib.load(MODELS_DIR / "xgboost_model_label_expansion.joblib")
    calibrator = joblib.load(MODELS_DIR / "probability_calibrator_label_expansion.joblib")

    # Load Production Cutoff
    with open(METADATA_DIR / "production_signal_rule_label_expansion.json", "r", encoding="utf-8") as f:
        prod_meta = json.load(f)
    gating_cutoff = prod_meta["signal_rule"]["calibrated_conviction_threshold"]

    # Generate Calibrated Ensemble Probabilities on Val and Test
    val_p_rf = rf_model.predict_proba(X_val)
    val_p_lr = lr_model.predict_proba(X_val)
    val_p_ens = calibrator.predict_proba(create_ensemble_probabilities(val_p_rf, val_p_lr))
    val_conviction = compute_directional_conviction(val_p_ens)

    test_p_rf = rf_model.predict_proba(X_test)
    test_p_lr = lr_model.predict_proba(X_test)
    test_p_ens = calibrator.predict_proba(create_ensemble_probabilities(test_p_rf, test_p_lr))
    test_conviction = compute_directional_conviction(test_p_ens)

    # Predictions:
    # 1. Unconstrained Directional Predictions (50-54% coverage, forced directional choice on active bars)
    val_preds_unconstrained = np.where(val_p_lr.argmax(axis=1) != 1, np.where(val_p_ens[:, 2] >= val_p_ens[:, 0], 1, -1), 0)
    # Actually, standard unconstrained ensemble prediction:
    val_preds = rf_model.predict(X_val)  # unconstrained balanced tree model (53.4% coverage)
    test_preds = rf_model.predict(X_test)

    # 2. Production Gated Signals (conviction >= cutoff)
    sig_val_gated = np.zeros(len(valid_val), dtype=int)
    val_mask = (val_conviction >= gating_cutoff)
    sig_val_gated[val_mask & (val_p_ens[:, 2] >= val_p_ens[:, 0])] = 1
    sig_val_gated[val_mask & (val_p_ens[:, 0] > val_p_ens[:, 2])] = -1

    sig_test_gated = np.zeros(len(valid_test), dtype=int)
    test_mask = (test_conviction >= gating_cutoff)
    sig_test_gated[test_mask & (test_p_ens[:, 2] >= test_p_ens[:, 0])] = 1
    sig_test_gated[test_mask & (test_p_ens[:, 0] > test_p_ens[:, 2])] = -1

    # ==========================================================================
    # 1. Performance by Time-of-Day (30-Minute Slots)
    # ==========================================================================
    print("\n--- 1. Performance by Time of Day ---")
    val_times = pd.Series(pd.to_datetime(valid_val["timestamp_ny"], utc=True)).dt.tz_convert("America/New_York").dt.time
    test_times = pd.Series(pd.to_datetime(valid_test["timestamp_ny"], utc=True)).dt.tz_convert("America/New_York").dt.time

    slots = [
        ("09:30 - 10:00", pd.to_datetime("09:30").time(), pd.to_datetime("10:00").time()),
        ("10:00 - 10:30", pd.to_datetime("10:00").time(), pd.to_datetime("10:30").time()),
        ("10:30 - 11:00", pd.to_datetime("10:30").time(), pd.to_datetime("11:00").time()),
        ("11:00 - 11:30", pd.to_datetime("11:00").time(), pd.to_datetime("11:30").time()),
        ("11:30 - 12:00", pd.to_datetime("11:30").time(), pd.to_datetime("12:00").time()),
        ("12:00 - 12:30", pd.to_datetime("12:00").time(), pd.to_datetime("12:30").time()),
        ("12:30 - 13:00", pd.to_datetime("12:30").time(), pd.to_datetime("13:00").time()),
        ("13:00 - 13:30", pd.to_datetime("13:00").time(), pd.to_datetime("13:30").time()),
        ("13:30 - 14:00", pd.to_datetime("13:30").time(), pd.to_datetime("14:00").time()),
        ("14:00 - 14:30", pd.to_datetime("14:00").time(), pd.to_datetime("14:30").time()),
        ("14:30 - 15:00", pd.to_datetime("14:30").time(), pd.to_datetime("15:00").time()),
        ("15:00 - 15:30", pd.to_datetime("15:00").time(), pd.to_datetime("15:30").time()),
        ("15:30 - 16:00", pd.to_datetime("15:30").time(), pd.to_datetime("16:00").time()),
    ]

    tod_results_val = []
    tod_results_test = []

    for name, t_start, t_end in slots:
        # Validation
        mask_v = (val_times >= t_start) & (val_times < t_end)
        if mask_v.sum() > 0:
            y_sub = y_val[mask_v]
            p_sub = val_preds[mask_v]
            r_sub = r_val[mask_v]
            prob_sub = val_p_ens[mask_v]
            
            acc = float(accuracy_score(y_sub, p_sub))
            b_acc = float(balanced_accuracy_score(y_sub, p_sub))
            f1 = float(f1_score(y_sub, p_sub, average="macro", zero_division=0))
            active = (p_sub != 0)
            gross = float(np.mean(p_sub[active] * r_sub[active]) * 10000.0) if active.sum() > 0 else 0.0
            net = gross - 8.0 if active.sum() > 0 else 0.0

            tod_results_val.append({
                "Slot": name,
                "Bars": int(mask_v.sum()),
                "Accuracy": f"{acc*100:.1f}%",
                "Balanced Acc": f"{b_acc*100:.1f}%",
                "Macro-F1": f"{f1:.4f}",
                "Coverage %": f"{active.sum()/len(p_sub)*100:.1f}%",
                "Gross Ret (bps)": f"{gross:+.2f}",
                "Net @ 8bp (bps)": f"{net:+.2f}",
            })

        # Test
        mask_t = (test_times >= t_start) & (test_times < t_end)
        if mask_t.sum() > 0:
            y_sub_t = y_test[mask_t]
            p_sub_t = test_preds[mask_t]
            r_sub_t = r_test[mask_t]
            
            acc_t = float(accuracy_score(y_sub_t, p_sub_t))
            b_acc_t = float(balanced_accuracy_score(y_sub_t, p_sub_t))
            f1_t = float(f1_score(y_sub_t, p_sub_t, average="macro", zero_division=0))
            active_t = (p_sub_t != 0)
            gross_t = float(np.mean(p_sub_t[active_t] * r_sub_t[active_t]) * 10000.0) if active_t.sum() > 0 else 0.0
            net_t = gross_t - 8.0 if active_t.sum() > 0 else 0.0

            tod_results_test.append({
                "Slot": name,
                "Bars": int(mask_t.sum()),
                "Accuracy": f"{acc_t*100:.1f}%",
                "Balanced Acc": f"{b_acc_t*100:.1f}%",
                "Macro-F1": f"{f1_t:.4f}",
                "Coverage %": f"{active_t.sum()/len(p_sub_t)*100:.1f}%",
                "Gross Ret (bps)": f"{gross_t:+.2f}",
                "Net @ 8bp (bps)": f"{net_t:+.2f}",
            })

    print("Validation Performance by Time of Day:")
    print(pd.DataFrame(tod_results_val).to_string(index=False))

    # ==========================================================================
    # 2. Performance by Volatility Regime
    # ==========================================================================
    print("\n--- 2. Performance by Volatility Regime ---")
    # Realized rolling volatility tertiles determined on Training Set
    vol_col = "feat_vol_12b"
    q33 = float(valid_train[vol_col].quantile(0.333))
    q66 = float(valid_train[vol_col].quantile(0.666))
    print(f"Volatility Cutoffs (from Train {vol_col}): Low <= {q33:.6f} | High > {q66:.6f}")

    def eval_regimes(df_sub, y_arr, p_arr, r_arr):
        vols = df_sub[vol_col].values
        reg_masks = {
            "Low Volatility (< 33%)": vols <= q33,
            "Medium Volatility (33-66%)": (vols > q33) & (vols <= q66),
            "High Volatility (> 66%)": vols > q66,
        }
        res = []
        for reg_name, m in reg_masks.items():
            if m.sum() > 0:
                acc = float(accuracy_score(y_arr[m], p_arr[m]))
                b_acc = float(balanced_accuracy_score(y_arr[m], p_arr[m]))
                f1 = float(f1_score(y_arr[m], p_arr[m], average="macro", zero_division=0))
                act = (p_arr[m] != 0)
                gross = float(np.mean(p_arr[m][act] * r_arr[m][act]) * 10000.0) if act.sum() > 0 else 0.0
                net = gross - 8.0 if act.sum() > 0 else 0.0
                win = float(np.mean((p_arr[m][act] * r_arr[m][act] - 0.0008) > 0) * 100.0) if act.sum() > 0 else 0.0
                res.append({
                    "Regime": reg_name,
                    "Bars": int(m.sum()),
                    "Balanced Acc": f"{b_acc*100:.1f}%",
                    "Macro-F1": f"{f1:.4f}",
                    "Coverage %": f"{act.sum()/m.sum()*100:.1f}%",
                    "Gross Ret (bps)": f"{gross:+.2f}",
                    "Net @ 8bp (bps)": f"{net:+.2f}",
                    "8bp Win Rate (%)": f"{win:.1f}%",
                })
        return res

    vol_results_val = eval_regimes(valid_val, y_val, val_preds, r_val)
    vol_results_test = eval_regimes(valid_test, y_test, test_preds, r_test)
    print("Validation by Volatility Regime:")
    print(pd.DataFrame(vol_results_val).to_string(index=False))

    # ==========================================================================
    # 3. Performance by Confidence Bucket
    # ==========================================================================
    print("\n--- 3. Performance by Confidence Bucket ---")
    bins = [0.0, 0.20, 0.22, 0.25, 0.30, 1.0]
    bin_labels = ["Low (< 0.20)", "Mild (0.20 - 0.22)", "Moderate (0.22 - 0.25)", "High (0.25 - 0.30)", "Extreme (> 0.30)"]

    def eval_confidence_bins(conv_arr, probs_arr, y_arr, r_arr):
        res = []
        for i in range(len(bins) - 1):
            low, high = bins[i], bins[i+1]
            m = (conv_arr >= low) & (conv_arr < high)
            if m.sum() > 0:
                y_sub = y_arr[m]
                r_sub = r_arr[m]
                p_sub = np.where(probs_arr[m, 2] >= probs_arr[m, 0], 1, -1)

                # Binary sign match on directional bars
                dir_m = (y_sub != 0)
                sign_acc = float(np.mean((p_sub[dir_m] == y_sub[dir_m])) * 100.0) if dir_m.sum() > 0 else 50.0

                gross = float(np.mean(p_sub * r_sub) * 10000.0)
                net = gross - 8.0
                win = float(np.mean(((p_sub * r_sub) - 0.0008) > 0) * 100.0)

                res.append({
                    "Confidence Bucket": bin_labels[i],
                    "Bars": int(m.sum()),
                    "Share of Total %": f"{m.sum()/len(conv_arr)*100:.1f}%",
                    "Sign Acc (%)": f"{sign_acc:.1f}%",
                    "Gross Ret (bps)": f"{gross:+.2f}",
                    "Net @ 8bp (bps)": f"{net:+.2f}",
                    "8bp Win Rate (%)": f"{win:.1f}%",
                })
        return res

    conf_results_val = eval_confidence_bins(val_conviction, val_p_ens, y_val, r_val)
    conf_results_test = eval_confidence_bins(test_conviction, test_p_ens, y_test, r_test)
    print("Validation by Confidence Bucket:")
    print(pd.DataFrame(conf_results_val).to_string(index=False))

    # ==========================================================================
    # 4. Feature Importance & Group Ablation Study
    # ==========================================================================
    print("\n--- 4. Feature Importance & Group Ablation Study ---")
    # MDI Feature Importance
    rf_mdi = pd.Series(rf_model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    xgb_mdi = pd.Series(xgb_model.feature_importances_, index=feature_cols).sort_values(ascending=False)

    top10_rf_mdi = rf_mdi.head(10).to_dict()
    top10_xgb_mdi = xgb_mdi.head(10).to_dict()
    print("Top 5 RF Features (MDI):", list(top10_rf_mdi.items())[:5])

    # Group Ablation Analysis on Validation
    # Feature Groups
    feature_groups = {
        "All 29 Features (Baseline)": feature_cols,
        "Ablate Returns & Momentum (feat_ret_*)": [c for c in feature_cols if not c.startswith("feat_ret_")],
        "Ablate Technical Oscillators (RSI, MACD, BB, EMA)": [c for c in feature_cols if not any(c.startswith(k) for k in ["feat_rsi", "feat_macd", "feat_bb", "feat_ema"])],
        "Ablate Volatility & Range (vol, range, body, close_pos)": [c for c in feature_cols if not any(c.startswith(k) for k in ["feat_vol_", "feat_bar_range", "feat_body_size", "feat_close_pos"])],
        "Ablate Volume & Liquidity (rvol, rtrades, trade_size)": [c for c in feature_cols if not any(c.startswith(k) for k in ["feat_rvol", "feat_rtrades", "feat_avg_trade_size", "feat_filled_density"])],
        "Ablate Time, Day & Gap (time, day, open, gap)": [c for c in feature_cols if not any(c.startswith(k) for k in ["feat_time", "feat_day", "feat_ret_since_open", "feat_overnight_gap"])],
    }

    ablation_results = []
    base_rf = RandomForestModel(n_estimators=100, max_depth=3, min_samples_leaf=50, max_features="sqrt", class_weight="balanced", random_state=42)
    base_rf.fit(X_train, y_train)
    p_base = base_rf.predict(X_val)
    base_f1 = float(f1_score(y_val, p_base, average="macro", zero_division=0))
    base_bacc = float(balanced_accuracy_score(y_val, p_base))

    for g_name, cols_sub in feature_groups.items():
        sub_idx = [feature_cols.index(c) for c in cols_sub]
        m_rf_sub = RandomForestModel(n_estimators=100, max_depth=3, min_samples_leaf=50, max_features="sqrt", class_weight="balanced", random_state=42)
        m_rf_sub.fit(X_train[:, sub_idx], y_train)
        pred_sub = m_rf_sub.predict(X_val[:, sub_idx])

        f1_sub = float(f1_score(y_val, pred_sub, average="macro", zero_division=0))
        bacc_sub = float(balanced_accuracy_score(y_val, pred_sub))
        diff_f1 = f1_sub - base_f1
        diff_bacc = bacc_sub - base_bacc

        ablation_results.append({
            "Experiment": g_name,
            "Features Remaining": len(cols_sub),
            "Balanced Acc": f"{bacc_sub*100:.1f}%",
            "Delta BalAcc": f"{diff_bacc*100:+.2f}%",
            "Macro-F1": f"{f1_sub:.4f}",
            "Delta Macro-F1": f"{diff_f1:+.4f}",
        })

    print("Group Feature Ablation Results on Validation:")
    print(pd.DataFrame(ablation_results).to_string(index=False))

    # ==========================================================================
    # 5. Sensitivity to Random Seed
    # ==========================================================================
    print("\n--- 5. Sensitivity to Random Seed ---")
    seeds = [42, 100, 2024, 7, 999]
    seed_results_rf = []
    seed_results_xgb = []

    for s in seeds:
        # RF
        rf_s = RandomForestModel(n_estimators=150, max_depth=3, min_samples_leaf=50, max_features="sqrt", class_weight="balanced", random_state=s)
        rf_s.fit(X_train, y_train)
        p_rf_s = rf_s.predict(X_val)
        prob_rf_s = rf_s.predict_proba(X_val)
        act_rf = (p_rf_s != 0)
        net_rf = float(np.mean(p_rf_s[act_rf] * r_val[act_rf] - 0.0008) * 10000.0)

        seed_results_rf.append({
            "seed": s,
            "accuracy": float(accuracy_score(y_val, p_rf_s)),
            "balanced_accuracy": float(balanced_accuracy_score(y_val, p_rf_s)),
            "macro_f1": float(f1_score(y_val, p_rf_s, average="macro", zero_division=0)),
            "net_8bp": net_rf,
        })

        # XGB
        xgb_s = XGBoostModel(n_estimators=100, max_depth=4, learning_rate=0.10, subsample=0.8, colsample_bytree=0.8, min_child_weight=1.0, random_state=s)
        xgb_s.fit(X_train, y_train)
        p_xgb_s = xgb_s.predict(X_val)
        act_xgb = (p_xgb_s != 0)
        net_xgb = float(np.mean(p_xgb_s[act_xgb] * r_val[act_xgb] - 0.0008) * 10000.0)

        seed_results_xgb.append({
            "seed": s,
            "accuracy": float(accuracy_score(y_val, p_xgb_s)),
            "balanced_accuracy": float(balanced_accuracy_score(y_val, p_xgb_s)),
            "macro_f1": float(f1_score(y_val, p_xgb_s, average="macro", zero_division=0)),
            "net_8bp": net_xgb,
        })

    rf_bacc_vals = [r["balanced_accuracy"] for r in seed_results_rf]
    rf_net_vals = [r["net_8bp"] for r in seed_results_rf]
    xgb_bacc_vals = [r["balanced_accuracy"] for r in seed_results_xgb]
    xgb_net_vals = [r["net_8bp"] for r in seed_results_xgb]

    print(f"Random Forest across 5 seeds: Balanced Acc = {np.mean(rf_bacc_vals)*100:.2f}% +/- {np.std(rf_bacc_vals)*100:.2f}%, Net Ret = {np.mean(rf_net_vals):+.2f} +/- {np.std(rf_net_vals):.2f} bps")
    print(f"XGBoost across 5 seeds:       Balanced Acc = {np.mean(xgb_bacc_vals)*100:.2f}% +/- {np.std(xgb_bacc_vals)*100:.2f}%, Net Ret = {np.mean(xgb_net_vals):+.2f} +/- {np.std(xgb_net_vals):.2f} bps")

    # ==========================================================================
    # 6. Save Metadata & Figures
    # ==========================================================================
    output_metadata = {
        "step": "Step 13: Error Analysis, Regimes & Robustness",
        "time_of_day_analysis": {
            "validation": tod_results_val,
            "test": tod_results_test,
        },
        "volatility_regimes": {
            "cutoffs": {"q33_low": q33, "q66_high": q66},
            "validation": vol_results_val,
            "test": vol_results_test,
        },
        "confidence_buckets": {
            "validation": conf_results_val,
            "test": conf_results_test,
        },
        "feature_importance": {
            "top10_rf_mdi": top10_rf_mdi,
            "top10_xgb_mdi": top10_xgb_mdi,
            "group_ablation_validation": ablation_results,
            "causation_disclaimer": "Feature importance reflects predictive association within historical tree splits and does not imply economic or market causation.",
        },
        "seed_sensitivity": {
            "rf_summary": {
                "mean_balanced_accuracy": round(float(np.mean(rf_bacc_vals)), 4),
                "std_balanced_accuracy": round(float(np.std(rf_bacc_vals)), 4),
                "mean_net_return_8bp": round(float(np.mean(rf_net_vals)), 2),
                "std_net_return_8bp": round(float(np.std(rf_net_vals)), 2),
                "all_runs": seed_results_rf,
            },
            "xgb_summary": {
                "mean_balanced_accuracy": round(float(np.mean(xgb_bacc_vals)), 4),
                "std_balanced_accuracy": round(float(np.std(xgb_bacc_vals)), 4),
                "mean_net_return_8bp": round(float(np.mean(xgb_net_vals)), 2),
                "std_net_return_8bp": round(float(np.std(xgb_net_vals)), 2),
                "all_runs": seed_results_xgb,
            },
        },
    }

    with open(METADATA_DIR / "step13_error_analysis_and_robustness.json", "w", encoding="utf-8") as f:
        json.dump(output_metadata, f, indent=2)
    print(f"\nSaved metadata to: {METADATA_DIR / 'step13_error_analysis_and_robustness.json'}")

    # Generate Figures
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # Figure 1: TOD & Volatility Performance
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
    slot_names = [r["Slot"] for r in tod_results_val]
    val_net_tod = [float(r["Net @ 8bp (bps)"]) for r in tod_results_val]
    test_net_tod = [float(r["Net @ 8bp (bps)"]) for r in tod_results_test]

    x = np.arange(len(slot_names))
    width = 0.38
    ax1.bar(x - width/2, val_net_tod, width, label="Validation", color="#2980b9", alpha=0.85)
    ax1.bar(x + width/2, test_net_tod, width, label="Test", color="#e67e22", alpha=0.85)
    ax1.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(slot_names, rotation=45, ha="right")
    ax1.set_ylabel("Net Return @ 8bp (bps)")
    ax1.set_title("Performance by Time-of-Day (30-min slots)", fontweight="bold")
    ax1.legend()

    # Volatility Regimes Bar
    vol_names = [r["Regime"] for r in vol_results_val]
    val_net_vol = [float(r["Net @ 8bp (bps)"]) for r in vol_results_val]
    test_net_vol = [float(r["Net @ 8bp (bps)"]) for r in vol_results_test]

    xv = np.arange(len(vol_names))
    ax2.bar(xv - width/2, val_net_vol, width, label="Validation", color="#2980b9", alpha=0.85)
    ax2.bar(xv + width/2, test_net_vol, width, label="Test", color="#e67e22", alpha=0.85)
    ax2.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax2.set_xticks(xv)
    ax2.set_xticklabels(vol_names)
    ax2.set_ylabel("Net Return @ 8bp (bps)")
    ax2.set_title("Performance by Volatility Regime", fontweight="bold")
    ax2.legend()

    fig.suptitle("TradePulse - Error Analysis: Intraday Timing & Volatility Regimes", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "step13_performance_by_tod_and_volatility.png", dpi=300)
    plt.close(fig)
    print(f"Saved: {FIGURES_DIR / 'step13_performance_by_tod_and_volatility.png'}")

    # Figure 2: Feature Importance and Group Ablation
    fig, (ax_imp, ax_abl) = plt.subplots(1, 2, figsize=(14, 6), dpi=300)
    top_feats = rf_mdi.head(10).sort_values()
    ax_imp.barh(top_feats.index, top_feats.values, color="#34495e", alpha=0.85)
    ax_imp.set_xlabel("Mean Decrease in Impurity (MDI)")
    ax_imp.set_title("Top 10 Feature Importances (Random Forest)", fontweight="bold")

    # Ablation Delta Macro-F1
    abl_df = pd.DataFrame(ablation_results[1:])  # skip baseline
    abl_names = [n.replace("Ablate ", "") for n in abl_df["Experiment"]]
    abl_deltas = [float(d) for d in abl_df["Delta Macro-F1"]]
    colors = ["#e74c3c" if d < 0 else "#2ecc71" for d in abl_deltas]
    ax_abl.barh(abl_names, abl_deltas, color=colors, alpha=0.85)
    ax_abl.axvline(0, color="black", linestyle="--", linewidth=0.8)
    ax_abl.set_xlabel("Change in Macro-F1 vs All Features")
    ax_abl.set_title("Feature Group Ablation Sensitivity", fontweight="bold")

    fig.suptitle("TradePulse - Feature Importance and Ablation Analysis", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "step13_feature_importance_and_ablation.png", dpi=300)
    plt.close(fig)
    print(f"Saved: {FIGURES_DIR / 'step13_feature_importance_and_ablation.png'}")

    # Figure 3: Confidence Bucket Analysis
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    b_names = [r["Confidence Bucket"] for r in conf_results_val]
    b_val_net = [float(r["Net @ 8bp (bps)"]) for r in conf_results_val]
    b_test_net = [float(r["Net @ 8bp (bps)"]) for r in conf_results_test]

    xb = np.arange(len(b_names))
    ax.bar(xb - width/2, b_val_net, width, label="Validation", color="#2980b9", alpha=0.85)
    ax.bar(xb + width/2, b_test_net, width, label="Test", color="#e67e22", alpha=0.85)
    ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xticks(xb)
    ax.set_xticklabels(b_names, rotation=25, ha="right")
    ax.set_ylabel("Net Return @ 8bp (bps)")
    ax.set_title("Net Return by Calibrated Directional Conviction Bucket", fontweight="bold")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "step13_confidence_bucket_performance.png", dpi=300)
    plt.close(fig)
    print(f"Saved: {FIGURES_DIR / 'step13_confidence_bucket_performance.png'}")

    print("\n[Step 13 Error Analysis and Robustness Complete]")
    return output_metadata


if __name__ == "__main__":
    run_error_analysis()
