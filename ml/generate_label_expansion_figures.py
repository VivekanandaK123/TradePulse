"""
TradePulse - Label Expansion Figures Generator
Generates high-resolution figures for Phase F deliverables:
1. label_expansion_class_distributions.png
2. label_expansion_reliability_curves.png
3. label_expansion_equity_curves.png
"""

import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.features import get_feature_column_names
from ml.label_expansion_experiments import apply_fixed_coverage_signals, compute_directional_conviction
from ml.gating import simulate_non_overlapping_trades
from ml.calibration import create_ensemble_probabilities

FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
METADATA_DIR = PROJECT_ROOT / "artifacts" / "metadata"
MODELS_DIR = PROJECT_ROOT / "artifacts" / "models"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.titlesize": 13,
})


def plot_class_distributions():
    """Figure 1: Class distributions across all 7 candidate labels."""
    train_df = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "train_features.csv")
    valid_train = train_df[train_df["valid_target"]].copy()

    labels = [
        ("label_fixed_20bp", "Fixed ±20bp\n(Original)"),
        ("label_fixed_15bp", "Fixed ±15bp\n(Selected)"),
        ("label_quantile_50", "Quantile 50%\n(±14.7bp)"),
        ("label_quantile_40", "Quantile 40%\n(±11.3bp)"),
        ("label_fixed_10bp", "Fixed ±10bp\n(Narrow)"),
        ("label_vol_k075", "Vol Dynamic\nk=0.75"),
        ("label_fixed_5bp", "Fixed ±5bp\n(Extreme)"),
    ]

    label_names = [name for _, name in labels]
    down_pcts, neut_pcts, up_pcts = [], [], []

    for col, _ in labels:
        counts = valid_train[col].value_counts(normalize=True).to_dict()
        down_pcts.append(counts.get(-1, 0.0) * 100.0)
        neut_pcts.append(counts.get(0, 0.0) * 100.0)
        up_pcts.append(counts.get(1, 0.0) * 100.0)

    x = np.arange(len(labels))
    width = 0.55

    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)
    bars_down = ax.bar(x, down_pcts, width, label="Down (-1)", color="#e74c3c", alpha=0.9)
    bars_neut = ax.bar(x, neut_pcts, width, bottom=down_pcts, label="Neutral (0)", color="#95a5a6", alpha=0.85)
    bottom_up = np.array(down_pcts) + np.array(neut_pcts)
    bars_up = ax.bar(x, up_pcts, width, bottom=bottom_up, label="Up (+1)", color="#2ecc71", alpha=0.9)

    ax.set_ylabel("Class Proportion (%)")
    ax.set_title("TradePulse - Class Balance Across 7 Candidate Label Definitions (Training Set, N=12,193)", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(label_names)
    ax.set_ylim(0, 105)
    ax.axhline(50, color="black", linestyle="--", alpha=0.4, linewidth=0.8)

    # Highlight selected label
    ax.patches[1].set_edgecolor("#2c3e50")
    ax.patches[1].set_linewidth(2)

    # Annotate neutral percentages
    for i in range(len(labels)):
        ax.text(x[i], down_pcts[i] + neut_pcts[i] / 2, f"{neut_pcts[i]:.1f}%\nNeut",
                ha="center", va="center", color="white", fontweight="bold", fontsize=9)

    ax.legend(loc="upper right", frameon=True)
    fig.tight_layout()
    out_path = FIGURES_DIR / "label_expansion_class_distributions.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_reliability_curves():
    """Figure 2: Reliability curves per class for raw vs calibrated Ensemble on fixed_15bp."""
    val_df = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "val_features.csv")
    valid_val = val_df[val_df["valid_target"]].copy().reset_index(drop=True)
    feature_cols = get_feature_column_names(valid_val)

    # Load models
    rf_model = joblib.load(MODELS_DIR / "random_forest_model_label_expansion.joblib")
    lr_model = joblib.load(MODELS_DIR / "logistic_regression_model_label_expansion.joblib")
    calibrator = joblib.load(MODELS_DIR / "probability_calibrator_label_expansion.joblib")

    val_X = valid_val[feature_cols].values
    val_y = valid_val["label_fixed_15bp"].values.astype(int)

    p_rf = rf_model.predict_proba(val_X)
    p_lr = lr_model.predict_proba(val_X)
    p_raw = create_ensemble_probabilities(p_rf, p_lr)
    p_cal = calibrator.predict_proba(p_raw)

    classes = [(-1, "Down (-1)", "#e74c3c"), (0, "Neutral (0)", "#7f8c8d"), (1, "Up (+1)", "#2ecc71")]
    class_idx_map = {-1: 0, 0: 1, 1: 2}

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), dpi=300, sharey=True)

    for i, (cls_val, cls_name, color) in enumerate(classes):
        ax = axes[i]
        c_idx = class_idx_map[cls_val]
        y_binary = (val_y == cls_val).astype(int)

        # Raw curve
        frac_pos_raw, mean_pred_raw = calibration_curve(y_binary, p_raw[:, c_idx], n_bins=8, strategy="uniform")
        # Calibrated curve
        frac_pos_cal, mean_pred_cal = calibration_curve(y_binary, p_cal[:, c_idx], n_bins=8, strategy="uniform")

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

    fig.suptitle("TradePulse - Probability Calibration Reliability Curves on Validation Set (label_fixed_15bp)",
                 fontsize=13, fontweight="bold", y=1.03)
    fig.tight_layout()
    out_path = FIGURES_DIR / "label_expansion_reliability_curves.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_equity_curves():
    """Figure 3: Cumulative net return equity curves across the target signal rate grid."""
    val_df = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "val_features.csv")
    valid_val = val_df[val_df["valid_target"]].copy().reset_index(drop=True)
    feature_cols = get_feature_column_names(valid_val)

    rf_model = joblib.load(MODELS_DIR / "random_forest_model_label_expansion.joblib")
    lr_model = joblib.load(MODELS_DIR / "logistic_regression_model_label_expansion.joblib")
    calibrator = joblib.load(MODELS_DIR / "probability_calibrator_label_expansion.joblib")

    val_X = valid_val[feature_cols].values
    p_rf = rf_model.predict_proba(val_X)
    p_lr = lr_model.predict_proba(val_X)
    p_raw = create_ensemble_probabilities(p_rf, p_lr)
    p_cal = calibrator.predict_proba(p_raw)

    rates = [
        ("Rate ~1 sig/day (1.3% cov)", 1.0 / 78.0, "#34495e"),
        ("Rate ~2 sig/day (2.6% cov)", 2.0 / 78.0, "#e67e22"),
        ("Rate ~5 sig/day (6.4% cov - Selected)", 5.0 / 78.0, "#2980b9"),
        ("Rate ~10 sig/day (12.8% cov)", 10.0 / 78.0, "#8e44ad"),
    ]

    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=300)

    n = len(valid_val)
    actual_returns = valid_val["target_return"].values
    session_dates = valid_val["session_date"].values

    # Simulate each rate
    for label, frac, color in rates:
        sigs, _ = apply_fixed_coverage_signals(p_cal, top_fraction=frac)
        sim = simulate_non_overlapping_trades(valid_val, sigs, horizon_bars=3, cost_bps=0.0008)
        
        # Build chronological trade points
        i = 0
        trade_indices = [0]
        cum_series = [0.0]
        curr_cum = 0.0
        
        while i < n:
            sig = sigs[i]
            if sig != 0:
                r = actual_returns[i]
                net = (sig * r) - 0.0008
                curr_cum += net * 100.0
                trade_indices.append(i + 3)
                cum_series.append(curr_cum)
                
                curr_date = session_dates[i]
                skip = 3
                for step in range(1, 4):
                    if i + step >= n or session_dates[i + step] != curr_date:
                        skip = step
                        break
                i += skip
            else:
                i += 1
                
        # Extend to end
        trade_indices.append(n - 1)
        cum_series.append(curr_cum)
            
        ax.step(trade_indices, cum_series, where="post", label=f"{label} (Final: {sim['cumulative_net_pct']:+.1f}%, DD: {sim['max_drawdown_pct']:.1f}%)",
                color=color, linewidth=2.0 if "Selected" in label else 1.4)

    ax.axhline(0, color="black", linestyle="--", alpha=0.5, linewidth=0.8)
    ax.set_xlabel("Validation Bar Index (Chronological, 5-min intervals)")
    ax.set_ylabel("Cumulative Net Return (%) after 8 bps Cost")
    ax.set_title("TradePulse - Realistic Non-Overlapping Backtest Equity Curves Across Signal Rates (Validation Set)", fontweight="bold")
    ax.legend(loc="lower left", frameon=True)
    fig.tight_layout()

    out_path = FIGURES_DIR / "label_expansion_equity_curves.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    print("Generating Label Expansion Visualizations...")
    plot_class_distributions()
    plot_reliability_curves()
    plot_equity_curves()
    print("All figures successfully created.")


if __name__ == "__main__":
    main()
