"""
TradePulse - Step 3: Target Definition and Label Generation

Implements pure pipeline logic for forward return computation and target labeling:
1. 3-bar forward return (15 minutes): R_{t, t+3} = close[t+3] / close[t] - 1.
2. Same-day boundary isolation: Never computes returns across midnight. Drops last 3 bars of each day.
3. Target quality tracking: Records whether the t+3 bar was synthesized (target_is_filled).
4. Causal thresholds:
   - Fixed cuts (0.10%, 0.15%, 0.20%)
   - Symmetric quantiles of |R| on training data (40th, 50th, 60th percentiles)
   - Causal volatility-adaptive band: k * sigma_{t, 3-bar} using only information known at t.
5. All thresholds are fit strictly on training data and frozen into artifacts/metadata/.
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, Union

import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.preprocessing import EXCHANGE_TIMEZONE

METADATA_DIR = PROJECT_ROOT / "artifacts" / "metadata"
FROZEN_THRESHOLDS_PATH = METADATA_DIR / "target_thresholds.json"


# ==============================================================================
# Forward Return Computation (Intraday Only)
# ==============================================================================

def compute_forward_returns(
    df: pd.DataFrame,
    horizon_bars: int = 3,
    vol_lookback_bars: int = 12,
) -> pd.DataFrame:
    """
    Computes 3-bar forward returns strictly within each trading session.

    - Drops the final `horizon_bars` (e.g. 3 bars) of each session (no same-day target).
    - Checks whether t+horizon was forward-filled (`target_is_filled`).
    - Calculates causal rolling volatility of 1-bar returns strictly up to time t.
    """
    out = df.copy()

    # Session date in America/New_York
    session_dates = (
        pd.to_datetime(out["timestamp"], utc=True)
        .dt.tz_convert(EXCHANGE_TIMEZONE)
        .dt.date
    )
    out["session_date"] = session_dates

    # 1. Forward close and is_filled at t + horizon_bars within the same day
    grouped_close = out.groupby("session_date")["close"]
    grouped_filled = out.groupby("session_date")["is_filled"]

    out["close_target"] = grouped_close.shift(-horizon_bars)
    out["target_is_filled"] = grouped_filled.shift(-horizon_bars).fillna(0).astype(int)

    # 2. Forward return over horizon
    out["target_return"] = (out["close_target"] / out["close"]) - 1.0

    # 3. Validity mask: Last 3 bars of every session will have NaN target
    out["valid_target"] = ~out["target_return"].isna()

    # 4. Causal 1-bar returns and rolling volatility known at time t (for volatility rule)
    out["ret_1bar"] = grouped_close.pct_change()
    # 12-bar rolling standard deviation (1 hour) of past 5-min returns
    out["causal_vol_12bar"] = (
        out.groupby("session_date")["ret_1bar"]
        .transform(lambda s: s.rolling(vol_lookback_bars, min_periods=max(3, vol_lookback_bars // 2)).std())
    )
    # Expected 3-bar volatility: sqrt(3) * 1-bar std
    out["causal_vol_3bar"] = np.sqrt(horizon_bars) * out["causal_vol_12bar"]

    return out


# ==============================================================================
# Candidate Threshold Computation (Training Data Only)
# ==============================================================================

def compute_candidate_thresholds(train_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Derives candidate threshold parameters strictly from the training dataset.

    Grid:
    1. Fixed cuts: 0.10%, 0.15%, 0.20% (all above ~0.05-0.10% round-trip costs).
    2. Symmetric quantile cuts of |R|: 40th, 50th, 60th percentiles.
    3. Causal volatility multipliers: k in {0.50, 0.75, 1.00} with median imputation
       for early session warm-up and a floor at 0.08% (cost safety).
    """
    valid_train = train_df[train_df["valid_target"]].copy()
    if valid_train.empty:
        raise ValueError("Training dataframe contains no valid target rows.")

    r = valid_train["target_return"]
    abs_r = r.abs()

    # Median causal volatility on train set to impute early morning warm-up
    median_vol_3bar = float(valid_train["causal_vol_3bar"].median())
    if np.isnan(median_vol_3bar):
        median_vol_3bar = float(r.std())

    threshold_params: Dict[str, Any] = {
        "fixed_rules": {
            "fixed_10bp": {"theta": 0.0010, "description": "Fixed cut at +/-0.10% (10 bps)"},
            "fixed_15bp": {"theta": 0.0015, "description": "Fixed cut at +/-0.15% (15 bps)"},
            "fixed_20bp": {"theta": 0.0020, "description": "Fixed cut at +/-0.20% (20 bps)"},
        },
        "quantile_rules": {
            "quantile_40": {
                "percentile": 40,
                "theta": float(abs_r.quantile(0.40)),
                "description": f"Symmetric 40th percentile of |R_train| ({abs_r.quantile(0.40)*100:.4f}%)",
            },
            "quantile_50": {
                "percentile": 50,
                "theta": float(abs_r.quantile(0.50)),
                "description": f"Symmetric 50th percentile (median) of |R_train| ({abs_r.quantile(0.50)*100:.4f}%)",
            },
            "quantile_60": {
                "percentile": 60,
                "theta": float(abs_r.quantile(0.60)),
                "description": f"Symmetric 60th percentile of |R_train| ({abs_r.quantile(0.60)*100:.4f}%)",
            },
        },
        "volatility_rules": {
            "vol_k050": {
                "k": 0.50,
                "median_vol_3bar": median_vol_3bar,
                "min_floor_theta": 0.0008,
                "description": "Causal volatility: theta_t = max(0.50 * sigma_{t,3b}, 0.08%)",
            },
            "vol_k075": {
                "k": 0.75,
                "median_vol_3bar": median_vol_3bar,
                "min_floor_theta": 0.0008,
                "description": "Causal volatility: theta_t = max(0.75 * sigma_{t,3b}, 0.08%)",
            },
            "vol_k100": {
                "k": 1.00,
                "median_vol_3bar": median_vol_3bar,
                "min_floor_theta": 0.0008,
                "description": "Causal volatility: theta_t = max(1.00 * sigma_{t,3b}, 0.08%)",
            },
        },
        "train_statistics": {
            "train_valid_bars": int(len(valid_train)),
            "return_mean": float(r.mean()),
            "return_std": float(r.std()),
            "abs_return_median": float(abs_r.median()),
            "cost_floor": 0.0008,
        },
    }

    # Save to artifacts/metadata/
    FROZEN_THRESHOLDS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(FROZEN_THRESHOLDS_PATH, "w", encoding="utf-8") as f:
        json.dump(threshold_params, f, indent=2)

    return threshold_params


# ==============================================================================
# Label Assignment (+1 Up, -1 Down, 0 Neutral)
# ==============================================================================

def assign_labels_for_rule(
    df: pd.DataFrame,
    rule_type: str,
    rule_name: str,
    threshold_params: Dict[str, Any],
) -> pd.Series:
    """
    Assigns ternary labels: Up (+1), Down (-1), Neutral (0) based on frozen threshold params.

    Supports:
    - rule_type="fixed": rule_name in ["fixed_10bp", "fixed_15bp", "fixed_20bp"]
    - rule_type="quantile": rule_name in ["quantile_40", "quantile_50", "quantile_60"]
    - rule_type="volatility": rule_name in ["vol_k050", "vol_k075", "vol_k100"]
    """
    if "target_return" not in df.columns:
        raise ValueError("Dataframe must contain 'target_return' column.")

    r = df["target_return"]

    if rule_type == "fixed":
        theta = float(threshold_params["fixed_rules"][rule_name]["theta"])
    elif rule_type == "quantile":
        theta = float(threshold_params["quantile_rules"][rule_name]["theta"])
    elif rule_type == "volatility":
        v_cfg = threshold_params["volatility_rules"][rule_name]
        k = float(v_cfg["k"])
        med_vol = float(v_cfg["median_vol_3bar"])
        floor = float(v_cfg["min_floor_theta"])

        # Use causal volatility computed at time t, imputed with train median if in warm-up
        vol = df["causal_vol_3bar"].fillna(med_vol)
        theta = np.maximum(k * vol, floor)
    else:
        raise ValueError(f"Unknown rule_type '{rule_type}'.")

    # Ternary classification: Up (+1), Down (-1), Neutral (0)
    labels = pd.Series(np.nan, index=df.index, dtype=float)
    valid_mask = df["valid_target"]

    if isinstance(theta, pd.Series):
        labels[valid_mask & (r > theta)] = 1.0
        labels[valid_mask & (r < -theta)] = -1.0
        labels[valid_mask & (r >= -theta) & (r <= theta)] = 0.0
    else:
        labels[valid_mask & (r > theta)] = 1.0
        labels[valid_mask & (r < -theta)] = -1.0
        labels[valid_mask & (r >= -theta) & (r <= theta)] = 0.0

    return labels


# ==============================================================================
# Pipeline Sanity Check Runner
# ==============================================================================

if __name__ == "__main__":
    print("[ml/targets.py] Running target pipeline validation...")
    from ml.preprocessing import PROCESSED_CSV_PATH

    clean_df = pd.read_csv(PROCESSED_CSV_PATH)
    with_returns = compute_forward_returns(clean_df, horizon_bars=3)

    train_mask = (with_returns["session_date"].astype(str) >= "2025-11-03") & (
        with_returns["session_date"].astype(str) <= "2026-06-30"
    )
    train_df = with_returns[train_mask].copy()

    params = compute_candidate_thresholds(train_df)
    print(f"  Candidate thresholds computed and frozen at: {FROZEN_THRESHOLDS_PATH}")

    # Test label generation
    labels_q50 = assign_labels_for_rule(train_df, "quantile", "quantile_50", params)
    print("  Quantile 50% class distribution on train:")
    print(labels_q50.value_counts(normalize=True).round(3).to_dict())
    print("  Target pipeline sanity check passed!")
