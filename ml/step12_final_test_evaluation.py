"""
TradePulse - Step 12: Final Test Evaluation (Once, Sealed Test Set)
Author: TradePulse Team (IIT Patna, IMPACT Lab)

Strict Protocol:
1. Touch the test set ONCE.
2. Evaluate frozen models, calibrator, and signal rules on test_features.csv (N = 3,525 valid bars, Sep 2 – Nov 6, 2026).
3. Compute full classification and economic metrics against all baselines with day-level block-bootstrap 95% CIs.
4. Run realistic non-overlapping backtest under 5, 8, 12 bps transaction costs.
5. Generate test artifacts, figures, and metadata.
"""

import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    log_loss,
    roc_auc_score,
)
from sklearn.calibration import calibration_curve
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.features import get_feature_column_names
from ml.evaluate import compute_classification_metrics
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
from ml.calibration import create_ensemble_probabilities
from ml.gating import simulate_non_overlapping_trades
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


def compute_economic_metrics_with_bootstrap(
    session_dates: pd.Series,
    signals: np.ndarray,
    returns: np.ndarray,
    cost_bps: float = 0.0008,
    n_bootstraps: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    active = (signals != 0)
    n_sig = int(np.sum(active))
    cov_pct = float(n_sig / len(signals) * 100.0)

    if n_sig == 0:
        return {
            "coverage_pct": 0.0,
            "n_signals": 0,
            "gross_bps": 0.0,
            "gross_95ci": (0.0, 0.0),
            "net_5bp_bps": 0.0,
            "net_8bp_bps": 0.0,
            "net_8bp_95ci": (0.0, 0.0),
            "net_12bp_bps": 0.0,
            "win_rate_8bp_pct": 0.0,
            "win_rate_95ci": (0.0, 0.0),
        }

    act_sig = signals[active]
    act_ret = returns[active]
    gross = act_sig * act_ret
    gross_bps = float(np.mean(gross) * 10000.0)
    net_5bp = float(np.mean(gross - 0.0005) * 10000.0)
    net_8bp = float(np.mean(gross - 0.0008) * 10000.0)
    net_12bp = float(np.mean(gross - 0.0012) * 10000.0)
    win_rate = float(np.mean((gross - 0.0008) > 0) * 100.0)

    # Day-level block bootstrap
    rng = np.random.default_rng(seed)
    unique_dates = np.unique(session_dates)
    n_days = len(unique_dates)

    df_b = pd.DataFrame({
        "date": session_dates.values,
        "sig": signals,
        "ret": returns,
    })
    day_groups = [g for _, g in df_b.groupby("date")]

    boot_gross = []
    boot_net_8bp = []
    boot_win = []

    for _ in range(n_bootstraps):
        idx = rng.choice(n_days, size=n_days, replace=True)
        sample_df = pd.concat([day_groups[i] for i in idx], ignore_index=True)
        s_act = (sample_df["sig"].values != 0)
        if np.sum(s_act) > 0:
            s_g = sample_df["sig"].values[s_act] * sample_df["ret"].values[s_act]
            boot_gross.append(np.mean(s_g) * 10000.0)
            boot_net_8bp.append(np.mean(s_g - 0.0008) * 10000.0)
            boot_win.append(np.mean((s_g - 0.0008) > 0) * 100.0)

    gross_ci = (round(float(np.percentile(boot_gross, 2.5)), 2), round(float(np.percentile(boot_gross, 97.5)), 2)) if boot_gross else (0.0, 0.0)
    net_ci = (round(float(np.percentile(boot_net_8bp, 2.5)), 2), round(float(np.percentile(boot_net_8bp, 97.5)), 2)) if boot_net_8bp else (0.0, 0.0)
    win_ci = (round(float(np.percentile(boot_win, 2.5)), 1), round(float(np.percentile(boot_win, 97.5)), 1)) if boot_win else (0.0, 0.0)

    return {
        "coverage_pct": round(cov_pct, 2),
        "n_signals": n_sig,
        "gross_bps": round(gross_bps, 2),
        "gross_95ci": gross_ci,
        "net_5bp_bps": round(net_5bp, 2),
        "net_8bp_bps": round(net_8bp, 2),
        "net_8bp_95ci": net_ci,
        "net_12bp_bps": round(net_12bp, 2),
        "win_rate_8bp_pct": round(win_rate, 1),
        "win_rate_95ci": win_ci,
    }


def evaluate_final_test_set():
    print("=" * 80)
    print("TradePulse - Step 12: Final Test Evaluation (Touching Test Set Once)")
    print("=" * 80)

    # 1. Load Data
    train_df = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "train_features.csv")
    test_df = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "test_features.csv")

    valid_train = train_df[train_df["valid_target"]].copy().reset_index(drop=True)
    valid_test = test_df[test_df["valid_target"]].copy().reset_index(drop=True)
    feature_cols = get_feature_column_names(valid_train)

    print(f"Loaded Train: {len(valid_train):,} bars | Test: {len(valid_test):,} bars across {valid_test['session_date'].nunique()} sessions")
    print(f"Target: {TARGET_COL} | Date Range: {valid_test['session_date'].min()} to {valid_test['session_date'].max()}")

    X_train = valid_train[feature_cols].values
    y_train = valid_train[TARGET_COL].values.astype(int)

    X_test = valid_test[feature_cols].values
    y_test = valid_test[TARGET_COL].values.astype(int)
    r_test = valid_test["target_return"].values
    d_test = valid_test["session_date"]
    t_test = valid_test["timestamp_ny"]

    # 2. Load Frozen Models
    print("\nLoading frozen models and calibrator...")
    lr_model = joblib.load(MODELS_DIR / "logistic_regression_model_label_expansion.joblib")
    rf_model = joblib.load(MODELS_DIR / "random_forest_model_label_expansion.joblib")
    xgb_model = joblib.load(MODELS_DIR / "xgboost_model_label_expansion.joblib")
    calibrator = joblib.load(MODELS_DIR / "probability_calibrator_label_expansion.joblib")

    # Load production rule
    with open(METADATA_DIR / "production_signal_rule_label_expansion.json", "r", encoding="utf-8") as f:
        prod_rule_meta = json.load(f)
    gating_cutoff = prod_rule_meta["signal_rule"]["calibrated_conviction_threshold"]
    print(f"Loaded production signal rule: threshold tau = {gating_cutoff}")

    # Fit baselines on training data
    rsi_col_idx = feature_cols.index("feat_rsi_14")
    m_maj = MajorityBaseline().fit(X_train, y_train)
    m_up = AlwaysUpBaseline().fit(X_train, y_train)
    m_mom = MomentumBaseline().fit(X_train, y_train)
    m_rsi = RSIBaseline(rsi_col_idx=rsi_col_idx).fit(X_train, y_train)
    m_tod = TimeOfDayBaseline().fit(valid_train["timestamp_ny"], y_train)

    # 3. Model Predictions on Test Set
    p_lr_test = lr_model.predict_proba(X_test)
    p_rf_test = rf_model.predict_proba(X_test)
    p_xgb_test = xgb_model.predict_proba(X_test)
    p_ens_raw = create_ensemble_probabilities(p_rf_test, p_lr_test)
    p_ens_cal = calibrator.predict_proba(p_ens_raw)
    p_tod_test = m_tod.predict_proba(t_test)

    pred_lr_test = lr_model.predict(X_test)
    pred_rf_test = rf_model.predict(X_test)
    pred_xgb_test = xgb_model.predict(X_test)
    pred_ens_test = np.array([-1, 0, 1])[np.argmax(p_ens_raw, axis=1)]

    pred_maj_test = m_maj.predict(X_test)
    pred_up_test = m_up.predict(X_test)
    pred_mom_test = m_mom.predict(X_test)
    pred_rsi_test = m_rsi.predict(X_test)
    pred_tod_test = m_tod.predict(t_test)

    # Production Gated Signals
    conviction_test = compute_directional_conviction(p_ens_cal)
    sig_prod_gated = np.zeros(len(valid_test), dtype=int)
    mask_gated = (conviction_test >= gating_cutoff)
    sig_prod_gated[mask_gated & (p_ens_cal[:, 2] >= p_ens_cal[:, 0])] = 1
    sig_prod_gated[mask_gated & (p_ens_cal[:, 0] > p_ens_cal[:, 2])] = -1

    # Daily signal frequencies for production rule
    daily_sigs = [int(np.sum(sig_prod_gated[d_test.values == d] != 0)) for d in np.unique(d_test)]
    avg_sigs_per_day = float(np.mean(daily_sigs))
    zero_days_pct = float(sum(1 for c in daily_sigs if c == 0) / len(daily_sigs) * 100.0)

    print(f"\nProduction Rule on Test Set: Observed Sigs/Day = {avg_sigs_per_day:.1f} | Zero-Signal Days = {zero_days_pct:.1f}%")

    # 4. Strategy & Baseline Suite Dictionary
    strategies = {
        "Majority (Always Neutral)": (pred_maj_test, None),
        "Always-Up (+1)": (pred_up_test, None),
        "Momentum (Sign R_t-1)": (pred_mom_test, None),
        "Basic RSI (30/70)": (pred_rsi_test, None),
        "Time-of-Day Baseline": (pred_tod_test, p_tod_test),
        "Logistic Regression (Unconstrained)": (pred_lr_test, p_lr_test),
        "Random Forest Tuned (Unconstrained)": (pred_rf_test, p_rf_test),
        "XGBoost Tuned (Unconstrained)": (pred_xgb_test, p_xgb_test),
        "Ensemble RF+LR (Unconstrained)": (pred_ens_test, p_ens_raw),
        "Production Rule (Calibrated Ensemble @ 5/day)": (sig_prod_gated, p_ens_cal),
    }

    # 5. Evaluate Full Metrics
    test_results_table = []
    full_test_details = {}

    for name, (signals, probs) in strategies.items():
        # Classification
        clf = compute_classification_metrics(y_test, signals, probs)
        # Economic with bootstrap
        econ = compute_economic_metrics_with_bootstrap(d_test, signals, r_test, cost_bps=0.0008)
        # Sign accuracy
        sign = compute_binary_sign_accuracy(y_test, probs) if probs is not None else {"sign_acc": 50.0, "ci_low": 50.0, "ci_high": 50.0}
        updown_auc = compute_up_down_roc_auc(y_test, probs) if probs is not None else None

        # Non-overlapping simulation
        sim = simulate_non_overlapping_trades(valid_test, signals, horizon_bars=3, cost_bps=0.0008)

        test_results_table.append({
            "Strategy": name,
            "Accuracy": f"{clf['accuracy']*100:.1f}%",
            "Balanced Acc": f"{clf['balanced_accuracy']*100:.1f}%",
            "Macro-F1": f"{clf['macro_f1']:.4f}",
            "Coverage %": f"{econ['coverage_pct']:.1f}%",
            "Signals / Day": f"{econ['n_signals'] / valid_test['session_date'].nunique():.1f}",
            "Gross Ret (bps)": f"{econ['gross_bps']:+.2f}",
            "Net @ 8bp (bps)": f"{econ['net_8bp_bps']:+.2f}",
            "Net 8bp 95% CI": str(econ['net_8bp_95ci']),
            "Win Rate (%)": f"{econ['win_rate_8bp_pct']:.1f}%",
            "Sign Acc (%)": f"{sign['sign_acc']:.1f}%",
            "Up-Down AUC": f"{updown_auc:.4f}" if updown_auc is not None else "N/A",
            "Non-Overlap Trades": sim["n_trades"],
            "Non-Overlap WinRate": f"{sim['win_rate_pct']:.1f}%",
            "Cum Net Return (%)": f"{sim['cumulative_net_pct']:+.2f}%",
            "Max Drawdown (%)": f"{sim['max_drawdown_pct']:.2f}%",
        })

        full_test_details[name] = {
            "classification": clf,
            "economic": econ,
            "sign_accuracy": sign,
            "updown_auc": updown_auc,
            "non_overlapping_backtest": sim,
        }

    df_test_summary = pd.DataFrame(test_results_table)
    print("\n" + "=" * 80)
    print("STEP 12: FINAL TEST SET PERFORMANCE TABLE (N = 3,525 valid bars, 47 sessions)")
    print("=" * 80)
    print(df_test_summary.to_string(index=False))

    # 6. Save Test Metadata
    test_artifact = {
        "evaluation_step": "Step 12: Final Test Set Evaluation",
        "protocol": "Sealed Test Set Touched Exactly Once",
        "dataset": {
            "symbol": "META",
            "partition": "test",
            "start_date": str(valid_test["session_date"].min()),
            "end_date": str(valid_test["session_date"].max()),
            "trading_days": int(valid_test["session_date"].nunique()),
            "total_bars": len(test_df),
            "valid_target_bars": len(valid_test),
            "class_distribution": valid_test[TARGET_COL].value_counts(normalize=True).to_dict(),
        },
        "target_label": TARGET_COL,
        "production_signal_rule": {
            "gating_cutoff": gating_cutoff,
            "observed_sigs_per_day": avg_sigs_per_day,
            "zero_signal_days_pct": zero_days_pct,
        },
        "summary_table": test_results_table,
        "detailed_results": full_test_details,
    }

    with open(METADATA_DIR / "step12_final_test_results.json", "w", encoding="utf-8") as f:
        json.dump(test_artifact, f, indent=2)
    print(f"\nSaved test metadata to: {METADATA_DIR / 'step12_final_test_results.json'}")

    # 7. Generate Step 12 Visualizations
    print("\nGenerating Step 12 visualizations...")
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # Figure 1: Test Equity Curves
    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=300)

    curves_to_plot = [
        ("Always-Up (+1)", pred_up_test, "#7f8c8d", 1.2),
        ("Momentum (Sign R_t-1)", pred_mom_test, "#e67e22", 1.2),
        ("Basic RSI (30/70)", pred_rsi_test, "#16a085", 1.2),
        ("Ensemble (Unconstrained)", pred_ens_test, "#9b59b6", 1.4),
        ("Production Rule (Calibrated Ensemble @ 5/day)", sig_prod_gated, "#2980b9", 2.2),
    ]

    n = len(valid_test)
    for label, sigs, color, lw in curves_to_plot:
        sim = simulate_non_overlapping_trades(valid_test, sigs, horizon_bars=3, cost_bps=0.0008)
        i = 0
        trade_indices = [0]
        cum_series = [0.0]
        curr_cum = 0.0

        while i < n:
            sig = sigs[i]
            if sig != 0:
                r = r_test[i]
                net = (sig * r) - 0.0008
                curr_cum += net * 100.0
                trade_indices.append(i + 3)
                cum_series.append(curr_cum)

                curr_date = d_test.iloc[i]
                skip = 3
                for step in range(1, 4):
                    if i + step >= n or d_test.iloc[i + step] != curr_date:
                        skip = step
                        break
                i += skip
            else:
                i += 1
        trade_indices.append(n - 1)
        cum_series.append(curr_cum)

        ax.step(trade_indices, cum_series, where="post",
                label=f"{label} (Final: {sim['cumulative_net_pct']:+.1f}%, DD: {sim['max_drawdown_pct']:.1f}%)",
                color=color, linewidth=lw)

    ax.axhline(0, color="black", linestyle="--", alpha=0.5, linewidth=0.8)
    ax.set_xlabel("Test Bar Index (Chronological, Sep 2 – Nov 6, 2026)")
    ax.set_ylabel("Cumulative Net Return (%) after 8 bps Cost")
    ax.set_title("TradePulse - Step 12: Final Out-of-Sample Test Equity Curves (Untouched Test Period)", fontweight="bold")
    ax.legend(loc="lower left", frameon=True)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "step12_test_equity_curves.png", dpi=300)
    plt.close(fig)
    print(f"Saved: {FIGURES_DIR / 'step12_test_equity_curves.png'}")

    # Figure 2: Calibration Curves on Test Set
    classes = [(-1, "Down (-1)", "#e74c3c"), (0, "Neutral (0)", "#7f8c8d"), (1, "Up (+1)", "#2ecc71")]
    class_idx_map = {-1: 0, 0: 1, 1: 2}

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), dpi=300, sharey=True)
    for i, (cls_val, cls_name, color) in enumerate(classes):
        ax = axes[i]
        c_idx = class_idx_map[cls_val]
        y_binary = (y_test == cls_val).astype(int)

        frac_pos_raw, mean_pred_raw = calibration_curve(y_binary, p_ens_raw[:, c_idx], n_bins=8, strategy="uniform")
        frac_pos_cal, mean_pred_cal = calibration_curve(y_binary, p_ens_cal[:, c_idx], n_bins=8, strategy="uniform")

        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect Calibration")
        ax.plot(mean_pred_raw, frac_pos_raw, "s-", color="#95a5a6", label="Raw Ensemble", alpha=0.8)
        ax.plot(mean_pred_cal, frac_pos_cal, "o-", color=color, linewidth=2, label="Platt Calibrated")

        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("Mean Predicted Probability")
        if i == 0:
            ax.set_ylabel("Empirical True Fraction")
        ax.set_title(f"Class: {cls_name}", fontweight="bold")
        ax.legend(loc="lower right" if cls_val != 0 else "upper left")

    fig.suptitle("TradePulse - Probability Calibration on Final Test Set (label_fixed_15bp)", fontsize=13, fontweight="bold", y=1.03)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "step12_test_calibration_curves.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {FIGURES_DIR / 'step12_test_calibration_curves.png'}")

    print("\n[Step 12 Final Test Execution Complete]")
    return test_artifact


if __name__ == "__main__":
    evaluate_final_test_set()

