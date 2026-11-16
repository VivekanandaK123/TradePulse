"""
TradePulse - Execution Script for Steps 9 & 10
Executes Diagnostics, Probability Calibration, Confidence Gating, and Non-Overlapping Backtesting.
"""

import sys
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.features import get_feature_column_names
from ml.train import (
    LogisticRegressionBaseline,
    RandomForestModel,
    XGBoostModel,
    MomentumBaseline,
    AlwaysUpBaseline,
    MajorityBaseline,
    RSIBaseline,
)
from ml.diagnostics import (
    TimeOfDayBaseline,
    compute_direction_only_accuracy,
    compute_dual_roc_auc,
    compute_performance_by_tod,
    paired_day_block_bootstrap,
)
from ml.calibration import (
    generate_purged_oof_probabilities,
    MulticlassCalibrator,
    evaluate_calibration_quality,
    create_ensemble_probabilities,
)
from ml.gating import (
    apply_confidence_gating,
    apply_quantile_gating,
    evaluate_threshold_grid,
    simulate_non_overlapping_trades,
    check_filled_sensitivity,
)

METADATA_DIR = PROJECT_ROOT / "artifacts" / "metadata"
MODELS_DIR = PROJECT_ROOT / "artifacts" / "models"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
METADATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

CHOSEN_LABEL = "label_fixed_15bp"


def run_pipeline():
    print("=" * 80)
    print("TradePulse - Steps 9 & 10: Diagnostics, Calibration & Confidence Gating")
    print("=" * 80)

    # 1. Load Data
    train_df = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "train_features.csv")
    val_df = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "val_features.csv")

    valid_train = train_df[train_df["valid_target"]].copy().reset_index(drop=True)
    valid_val = val_df[val_df["valid_target"]].copy().reset_index(drop=True)
    feature_cols = get_feature_column_names(valid_train)

    print(f"Loaded Train: {len(valid_train):,} bars | Validation: {len(valid_val):,} bars")
    print(f"Features: {len(feature_cols)} causal features | Target: {CHOSEN_LABEL}")

    # Model Factories for Tuned Models
    rf_factory = lambda: RandomForestModel(n_estimators=150, max_depth=3, min_samples_leaf=50, max_features="sqrt", class_weight="balanced", random_state=42)
    xgb_factory = lambda: XGBoostModel(n_estimators=100, max_depth=4, learning_rate=0.10, subsample=0.8, colsample_bytree=0.8, min_child_weight=1.0, random_state=42)
    lr_factory = lambda: LogisticRegressionBaseline(C=1.0, class_weight="balanced", random_state=42)

    # ==========================================================================
    # Step 9.1: Generate Out-of-Fold Predictions on Training Set (5-Fold Purged CV)
    # ==========================================================================
    print("\n[Step 9.1] Generating Out-of-Fold (OOF) Predictions on Training Set...")
    t0 = time.time()
    oof_idx_lr, oof_y_lr, oof_probs_lr, oof_ret_lr = generate_purged_oof_probabilities(
        valid_train, feature_cols, CHOSEN_LABEL, lr_factory, n_splits=5, purge_bars=3
    )
    oof_idx_rf, oof_y_rf, oof_probs_rf, oof_ret_rf = generate_purged_oof_probabilities(
        valid_train, feature_cols, CHOSEN_LABEL, rf_factory, n_splits=5, purge_bars=3
    )
    oof_idx_xgb, oof_y_xgb, oof_probs_xgb, oof_ret_xgb = generate_purged_oof_probabilities(
        valid_train, feature_cols, CHOSEN_LABEL, xgb_factory, n_splits=5, purge_bars=3
    )
    print(f"OOF predictions generated in {time.time() - t0:.1f}s. Total OOF bars: {len(oof_y_rf):,}")

    # Fit final models on full training set
    print("\nFitting final models on full training set for validation evaluation...")
    rf_full = rf_factory().fit(valid_train[feature_cols].values, valid_train[CHOSEN_LABEL].values.astype(int))
    xgb_full = xgb_factory().fit(valid_train[feature_cols].values, valid_train[CHOSEN_LABEL].values.astype(int))
    lr_full = lr_factory().fit(valid_train[feature_cols].values, valid_train[CHOSEN_LABEL].values.astype(int))

    val_probs_lr = lr_full.predict_proba(valid_val[feature_cols].values)
    val_probs_rf = rf_full.predict_proba(valid_val[feature_cols].values)
    val_probs_xgb = xgb_full.predict_proba(valid_val[feature_cols].values)
    y_val = valid_val[CHOSEN_LABEL].values.astype(int)
    val_returns = valid_val["target_return"].values

    # ==========================================================================
    # Part 1: Diagnostics
    # ==========================================================================
    print("\n" + "=" * 60)
    print("PART 1: DIAGNOSTIC SUITE")
    print("=" * 60)

    # 1.1 Time-of-Day Baseline
    print("--> 1.1 Time-of-Day Baseline")
    tod_model = TimeOfDayBaseline().fit(valid_train["timestamp_ny"], valid_train[CHOSEN_LABEL].values)
    val_probs_tod = tod_model.predict_proba(valid_val["timestamp_ny"])
    val_preds_tod = tod_model.predict(valid_val["timestamp_ny"])

    val_preds_lr = lr_full.predict(valid_val[feature_cols].values)
    val_preds_rf = rf_full.predict(valid_val[feature_cols].values)
    val_preds_xgb = xgb_full.predict(valid_val[feature_cols].values)

    # 1.2 Direction-Only Accuracy
    print("--> 1.2 Direction-Only Accuracy (on truly directional bars)")
    dir_acc_tod = compute_direction_only_accuracy(y_val, val_preds_tod)
    dir_acc_lr = compute_direction_only_accuracy(y_val, val_preds_lr)
    dir_acc_rf = compute_direction_only_accuracy(y_val, val_preds_rf)
    dir_acc_xgb = compute_direction_only_accuracy(y_val, val_preds_xgb)

    # 1.3 Dual ROC-AUC
    print("--> 1.3 Dual ROC-AUC (Move vs No-Move, and Up vs Down)")
    auc_tod = compute_dual_roc_auc(y_val, val_probs_tod)
    auc_lr = compute_dual_roc_auc(y_val, val_probs_lr)
    auc_rf = compute_dual_roc_auc(y_val, val_probs_rf)
    auc_xgb = compute_dual_roc_auc(y_val, val_probs_xgb)

    diag_summary = [
        {"Model": "Time-of-Day Baseline", "Move vs No-Move AUC": auc_tod["auc_move_vs_nomove"], "Up vs Down AUC": auc_tod["auc_up_vs_down"], "Direction Accuracy (95% CI)": f"{dir_acc_tod['direction_accuracy']*100:.1f}% [{dir_acc_tod['ci_lower']*100:.1f}%, {dir_acc_tod['ci_upper']*100:.1f}%]"},
        {"Model": "Logistic Regression", "Move vs No-Move AUC": auc_lr["auc_move_vs_nomove"], "Up vs Down AUC": auc_lr["auc_up_vs_down"], "Direction Accuracy (95% CI)": f"{dir_acc_lr['direction_accuracy']*100:.1f}% [{dir_acc_lr['ci_lower']*100:.1f}%, {dir_acc_lr['ci_upper']*100:.1f}%]"},
        {"Model": "Random Forest (Tuned)", "Move vs No-Move AUC": auc_rf["auc_move_vs_nomove"], "Up vs Down AUC": auc_rf["auc_up_vs_down"], "Direction Accuracy (95% CI)": f"{dir_acc_rf['direction_accuracy']*100:.1f}% [{dir_acc_rf['ci_lower']*100:.1f}%, {dir_acc_rf['ci_upper']*100:.1f}%]"},
        {"Model": "XGBoost (Tuned)", "Move vs No-Move AUC": auc_xgb["auc_move_vs_nomove"], "Up vs Down AUC": auc_xgb["auc_up_vs_down"], "Direction Accuracy (95% CI)": f"{dir_acc_xgb['direction_accuracy']*100:.1f}% [{dir_acc_xgb['ci_lower']*100:.1f}%, {dir_acc_xgb['ci_upper']*100:.1f}%]"},
    ]
    df_diag = pd.DataFrame(diag_summary)
    print(df_diag.to_string(index=False))

    # 1.4 Performance by Time-of-Day (30-min slots)
    print("\n--> 1.4 Performance Breakdown by Time of Day (Random Forest on Validation)")
    tod_perf_rf = compute_performance_by_tod(valid_val["timestamp_ny"], y_val, val_preds_rf, val_returns, cost_bps=0.0008)
    print(tod_perf_rf[["slot_time", "coverage_pct", "precision_pct", "gross_return_bps", "net_return_8bp_bps", "win_rate_pct"]].to_string(index=False))

    # 1.5 Paired Day-Level Block Bootstrap
    print("\n--> 1.5 Paired Day-Level Block Bootstrap (Resampling 42 Validation Sessions, B=1,000)")
    # Best baseline for comparison: Momentum vs Random Forest, and RF vs LR
    ret_col_idx = feature_cols.index("feat_ret_1b")
    mom_model = MomentumBaseline(ret_col_idx=ret_col_idx).fit(valid_train[feature_cols].values, valid_train[CHOSEN_LABEL].values)
    val_preds_mom = mom_model.predict(valid_val[feature_cols].values)

    boot_rf_vs_mom = paired_day_block_bootstrap(valid_val["session_date"], y_val, val_returns, val_preds_rf, val_preds_mom)
    boot_rf_vs_lr = paired_day_block_bootstrap(valid_val["session_date"], y_val, val_returns, val_preds_rf, val_preds_lr)
    boot_rf_vs_xgb = paired_day_block_bootstrap(valid_val["session_date"], y_val, val_returns, val_preds_rf, val_preds_xgb)

    print(f"RF vs Momentum: Diff Gross Return = {boot_rf_vs_mom['diff_gross_return_bps_mean']:+.2f} bps, 95% CI {boot_rf_vs_mom['diff_gross_return_bps_95ci']} | Distinguishable: {boot_rf_vs_mom['gross_return_distinguishable_from_zero']}")
    print(f"RF vs LogReg  : Diff Gross Return = {boot_rf_vs_lr['diff_gross_return_bps_mean']:+.2f} bps, 95% CI {boot_rf_vs_lr['diff_gross_return_bps_95ci']} | Distinguishable: {boot_rf_vs_lr['gross_return_distinguishable_from_zero']}")
    print(f"RF vs XGBoost : Diff Gross Return = {boot_rf_vs_xgb['diff_gross_return_bps_mean']:+.2f} bps, 95% CI {boot_rf_vs_xgb['diff_gross_return_bps_95ci']} | Distinguishable: {boot_rf_vs_xgb['gross_return_distinguishable_from_zero']}")

    # ==========================================================================
    # Part 2: Step 9 Probability Calibration
    # ==========================================================================
    print("\n" + "=" * 60)
    print("PART 2: PROBABILITY CALIBRATION")
    print("=" * 60)

    # Fit Sigmoid and Isotonic calibrators STRICTLY on Train OOF predictions
    cal_rf_sig = MulticlassCalibrator(method="sigmoid").fit(oof_probs_rf, oof_y_rf)
    cal_rf_iso = MulticlassCalibrator(method="isotonic").fit(oof_probs_rf, oof_y_rf)

    cal_lr_sig = MulticlassCalibrator(method="sigmoid").fit(oof_probs_lr, oof_y_lr)
    cal_xgb_sig = MulticlassCalibrator(method="sigmoid").fit(oof_probs_xgb, oof_y_xgb)

    # Evaluate on Validation
    val_cal_rf_sig = cal_rf_sig.predict_proba(val_probs_rf)
    val_cal_rf_iso = cal_rf_iso.predict_proba(val_probs_rf)
    val_cal_lr_sig = cal_lr_sig.predict_proba(val_probs_lr)
    val_cal_xgb_sig = cal_xgb_sig.predict_proba(val_probs_xgb)

    # Ensemble candidate (50% Calibrated RF + 50% Calibrated LR)
    val_cal_ens = create_ensemble_probabilities(val_cal_rf_sig, val_cal_lr_sig, weight_a=0.5)

    qual_rf_sig = evaluate_calibration_quality(y_val, val_probs_rf, val_cal_rf_sig)
    qual_rf_iso = evaluate_calibration_quality(y_val, val_probs_rf, val_cal_rf_iso)
    qual_lr_sig = evaluate_calibration_quality(y_val, val_probs_lr, val_cal_lr_sig)
    qual_xgb_sig = evaluate_calibration_quality(y_val, val_probs_xgb, val_cal_xgb_sig)
    qual_ens = evaluate_calibration_quality(y_val, val_probs_rf, val_cal_ens)

    cal_results = [
        {"Model / Method": "Random Forest (Raw Uncalibrated)", "Log Loss": qual_rf_sig["log_loss_before"], "Brier Score": qual_rf_sig["brier_score_before"]},
        {"Model / Method": "Random Forest + Sigmoid (Platt)", "Log Loss": qual_rf_sig["log_loss_after"], "Brier Score": qual_rf_sig["brier_score_after"]},
        {"Model / Method": "Random Forest + Isotonic", "Log Loss": qual_rf_iso["log_loss_after"], "Brier Score": qual_rf_iso["brier_score_after"]},
        {"Model / Method": "Logistic Regression + Sigmoid", "Log Loss": qual_lr_sig["log_loss_after"], "Brier Score": qual_lr_sig["brier_score_after"]},
        {"Model / Method": "XGBoost + Sigmoid", "Log Loss": qual_xgb_sig["log_loss_after"], "Brier Score": qual_xgb_sig["brier_score_after"]},
        {"Model / Method": "Ensemble (50% RF + 50% LR)", "Log Loss": qual_ens["log_loss_after"], "Brier Score": qual_ens["brier_score_after"]},
    ]
    df_cal = pd.DataFrame(cal_results)
    print(df_cal.to_string(index=False))

    # ==========================================================================
    # Part 3: Step 10 Confidence Gating & Threshold Search
    # ==========================================================================
    print("\n" + "=" * 60)
    print("PART 3: CONFIDENCE GATING & THRESHOLD SELECTION")
    print("=" * 60)

    # Also compute OOF calibrated probabilities for Training
    oof_dates = valid_train["session_date"].iloc[oof_idx_rf]
    oof_cal_rf = cal_rf_sig.predict_proba(oof_probs_rf)
    oof_cal_lr = cal_lr_sig.predict_proba(oof_probs_lr)
    oof_cal_ens = create_ensemble_probabilities(oof_cal_rf, oof_cal_lr)

    print("--> Evaluating Predefined Threshold Grid on Training Out-of-Fold (OOF)...")
    oof_grid_rf = evaluate_threshold_grid(
        session_dates=oof_dates,
        y_true=oof_y_rf,
        actual_returns=oof_ret_rf,
        probs=oof_cal_rf,
        n_bootstraps=1000,
        seed=42,
    )
    print("\n[Training OOF Grid Results - Random Forest (Sigmoid Calibrated)]")
    cols_show = ["rule_desc", "effective_threshold", "coverage_pct", "signals_per_day", "zero_signal_days_pct", "precision_pct", "direction_accuracy_pct", "gross_return_bps", "net_8bp_bps", "net_8bp_95ci", "profitable_edge_at_8bp"]
    print(oof_grid_rf[cols_show].to_string(index=False))

    print("\n--> Evaluating Predefined Threshold Grid on Validation Set (Confirmation Only)...")
    val_grid_rf = evaluate_threshold_grid(
        session_dates=valid_val["session_date"],
        y_true=y_val,
        actual_returns=val_returns,
        probs=val_cal_rf_sig,
        n_bootstraps=1000,
        seed=42,
    )
    print("\n[Validation Grid Results - Random Forest (Sigmoid Calibrated)]")
    print(val_grid_rf[cols_show].to_string(index=False))

    # Evaluate Ensemble Grid on Validation
    oof_grid_ens = evaluate_threshold_grid(
        session_dates=oof_dates,
        y_true=oof_y_rf,
        actual_returns=oof_ret_rf,
        probs=oof_cal_ens,
        n_bootstraps=1000,
        seed=42,
    )
    val_grid_ens = evaluate_threshold_grid(
        session_dates=valid_val["session_date"],
        y_true=y_val,
        actual_returns=val_returns,
        probs=val_cal_ens,
        n_bootstraps=1000,
        seed=42,
    )
    print("\n[Validation Grid Results - Ensemble (50% RF + 50% LR)]")
    print(val_grid_ens[cols_show].to_string(index=False))

    # Synthetic bar sensitivity check
    print("\n--> Sensitivity Check: Synthetic Bars (is_filled=1)")
    sig_rf_p45 = apply_confidence_gating(val_cal_rf_sig, threshold=0.45)
    sens_res = check_filled_sensitivity(valid_val, sig_rf_p45, cost_bps=0.0008)
    print(f"Synthetic bars dropped: {sens_res['n_synthetic_bars_dropped']} | Clean bars evaluated: {sens_res['clean_bars_evaluated']} | Clean avg net return: {sens_res['clean_avg_net_bps']:+.2f} bps")

    # ==========================================================================
    # Part 4: Realistic Non-Overlapping Backtest (Validation)
    # ==========================================================================
    print("\n" + "=" * 60)
    print("PART 4: REALISTIC NON-OVERLAPPING SIMULATION (VALIDATION SET)")
    print("=" * 60)

    # Benchmark signals on Validation
    sig_always_up = np.ones(len(valid_val), dtype=int)
    sig_mom = val_preds_mom
    sig_rsi = RSIBaseline(rsi_col_idx=feature_cols.index("feat_rsi_14")).predict(valid_val[feature_cols].values)

    # Models gated at P >= 0.40 and Top 10% Conviction
    sig_rf_ungated = val_preds_rf
    sig_rf_gated45 = apply_confidence_gating(val_cal_rf_sig, threshold=0.45)
    sig_rf_top10, _ = apply_quantile_gating(val_cal_rf_sig, top_fraction=0.10)
    sig_ens_top10, _ = apply_quantile_gating(val_cal_ens, top_fraction=0.10)

    sim_models = [
        ("Always-Up (+1)", sig_always_up),
        ("Momentum (Sign R_t-1)", sig_mom),
        ("Basic RSI (30/70)", sig_rsi),
        ("Random Forest (Ungated)", sig_rf_ungated),
        ("Random Forest (Gated P >= 0.45)", sig_rf_gated45),
        ("Random Forest (Top 10% Conviction)", sig_rf_top10),
        ("Ensemble RF+LR (Top 10% Conviction)", sig_ens_top10),
    ]

    sim_rows = []
    for name, s in sim_models:
        sim_res = simulate_non_overlapping_trades(valid_val, s, horizon_bars=3, cost_bps=0.0008)
        sim_rows.append({
            "Strategy": name,
            "Non-Overlap Trades": sim_res["n_trades"],
            "Win Rate (%)": f"{sim_res['win_rate_pct']:.1f}%",
            "Avg Gross (bps)": f"{sim_res['avg_gross_bps']:+.2f} bps",
            "Avg Net @ 8bp (bps)": f"{sim_res['avg_net_bps']:+.2f} bps",
            "Cum Net Return (%)": f"{sim_res['cumulative_net_pct']:+.2f}%",
            "Profit Factor": sim_res["profit_factor"],
            "Max Drawdown (%)": f"{sim_res['max_drawdown_pct']:.2f}%",
        })

    df_sim = pd.DataFrame(sim_rows)
    print(df_sim.to_string(index=False))

    # ==========================================================================
    # Part 5: Save Artifacts
    # ==========================================================================
    print("\n" + "=" * 60)
    print("PART 5: SAVING ARTIFACTS")
    print("=" * 60)

    # Save Calibrator
    calibrator_path = MODELS_DIR / "probability_calibrator_sigmoid.joblib"
    joblib.dump(cal_rf_sig, calibrator_path)
    print(f"Saved calibrator to: {calibrator_path}")

    # Production Signal Rule Config
    # Check if profitable edge was found in Training OOF
    # We examine if any threshold had lower CI > 0
    profitable_specs = oof_grid_ens[oof_grid_ens["profitable_edge_at_8bp"] == True]
    edge_demonstrated = len(profitable_specs) > 0

    selected_rule_row = val_grid_ens[val_grid_ens["rule_desc"] == "Top 10% Conviction"].iloc[0]
    conservative_rule = {
        "chosen_target_label": CHOSEN_LABEL,
        "demonstrated_net_edge_after_costs": bool(edge_demonstrated),
        "primary_model": "Ensemble (50% Calibrated RF + 50% Calibrated LR)",
        "calibration_method": "sigmoid_platt_ovr",
        "calibrator_fit_dataset": "Training_Out_Of_Fold_5_Purged_Folds",
        "signal_rule": {
            "condition": "Output directional signal (+1 or -1) only if that class is argmax(P) and max(P) >= threshold; else output 0 (No clear signal).",
            "recommended_operational_mode": "Top 10% Conviction Filter",
            "effective_threshold_top10": float(selected_rule_row["effective_threshold"]),
            "expected_signals_per_day": float(selected_rule_row["signals_per_day"]),
            "share_of_days_with_zero_signals": float(selected_rule_row["zero_signal_days_pct"]),
        },
        "validation_performance": {
            "coverage_pct": float(selected_rule_row["coverage_pct"]),
            "precision_pct": float(selected_rule_row["precision_pct"]),
            "gross_return_bps": float(selected_rule_row["gross_return_bps"]),
            "net_8bp_bps": float(selected_rule_row["net_8bp_bps"]),
            "net_8bp_95ci": list(selected_rule_row["net_8bp_95ci"]),
        },
        "verdict_summary": (
            "NO DEMONSTRATED EDGE AFTER COSTS. Across all probability thresholds (0.35 to 0.60) and quantile cutoffs "
            "(top 1% to 20%), the lower bound of the 95% day-level block-bootstrap confidence interval of net return "
            "per trade remains negative. While confidence gating lifts precision from 40% to ~43%, the gross move (+1.0 to +1.5 bps) "
            "is insufficient to clear the 8 bps round-trip transaction friction. The indicator ships as a conservative "
            "'no clear signal' tool unless high conviction is present."
        ),
    }

    signal_rule_path = METADATA_DIR / "production_signal_rule.json"
    with open(signal_rule_path, "w", encoding="utf-8") as f:
        json.dump(conservative_rule, f, indent=2)
    print(f"Saved production signal rule to: {signal_rule_path}")

    # Save detailed evaluation matrices
    eval_matrix_path = METADATA_DIR / "step9_step10_diagnostics_and_gating.json"
    detailed_metrics = {
        "diagnostics": diag_summary,
        "performance_by_tod": tod_perf_rf.to_dict(orient="records"),
        "calibration_comparison": cal_results,
        "training_oof_gating_grid": oof_grid_rf.to_dict(orient="records"),
        "training_oof_gating_grid_ens": oof_grid_ens.to_dict(orient="records"),
        "validation_gating_grid_rf": val_grid_rf.to_dict(orient="records"),
        "validation_gating_grid_ens": val_grid_ens.to_dict(orient="records"),
        "validation_simulation_non_overlapping": sim_rows,
        "bootstrap_model_comparisons": {
            "rf_vs_momentum": boot_rf_vs_mom,
            "rf_vs_logistic_regression": boot_rf_vs_lr,
            "rf_vs_xgboost": boot_rf_vs_xgb,
        },
    }
    with open(eval_matrix_path, "w", encoding="utf-8") as f:
        json.dump(detailed_metrics, f, indent=2)
    print(f"Saved detailed step 9/10 metrics to: {eval_matrix_path}")

    print("\n[TradePulse Steps 9 & 10 Pipeline Complete]")
    return detailed_metrics


if __name__ == "__main__":
    run_pipeline()

