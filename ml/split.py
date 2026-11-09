"""
TradePulse - Step 4: Chronological Dataset Partitioning

Applies frozen calendar boundaries to partition the processed dataset into:
- train.csv: 2025-11-03 to 2026-06-30 (164 trading days, 12,720 bars)
- val.csv:   2026-07-02 to 2026-08-31 (42 trading days, 3,276 bars)
- test.csv:  2026-09-02 to 2026-11-06 (47 trading days, 3,666 bars)
- Embargo buffer days (2026-07-01 and 2026-09-01, 78 bars each) are strictly isolated.
"""

import sys
import json
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SPLIT_CONFIG_PATH = PROJECT_ROOT / "artifacts" / "metadata" / "split_config.json"
TARGETS_CSV_PATH = PROJECT_ROOT / "data" / "processed" / "META_5min_targets.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def partition_and_save_datasets(
    targets_csv_path: Path = TARGETS_CSV_PATH,
    config_path: Path = SPLIT_CONFIG_PATH,
    output_dir: Path = PROCESSED_DIR,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Partitions the dataset chronologically and saves train.csv, val.csv, and test.csv.
    """
    print(f"\n[TradePulse Step 4: Dataset Partitioning]")
    print(f"  Reading source targets dataset: {targets_csv_path}")
    df = pd.read_csv(targets_csv_path)

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)["splits"]

    train_cfg = cfg["train"]
    val_cfg = cfg["validation"]
    test_cfg = cfg["test"]

    # Extract slices by session date
    train_mask = (df["session_date"] >= train_cfg["start_date"]) & (
        df["session_date"] <= train_cfg["end_date"]
    )
    val_mask = (df["session_date"] >= val_cfg["start_date"]) & (
        df["session_date"] <= val_cfg["end_date"]
    )
    test_mask = (df["session_date"] >= test_cfg["start_date"]) & (
        df["session_date"] <= test_cfg["end_date"]
    )

    train_df = df[train_mask].copy().reset_index(drop=True)
    val_df = df[val_mask].copy().reset_index(drop=True)
    test_df = df[test_mask].copy().reset_index(drop=True)

    # Sanity checks against frozen calendar specs
    assert len(train_df) == train_cfg["total_bars"], (
        f"Train bar count mismatch: {len(train_df)} vs {train_cfg['total_bars']}"
    )
    assert len(val_df) == val_cfg["total_bars"], (
        f"Val bar count mismatch: {len(val_df)} vs {val_cfg['total_bars']}"
    )
    assert len(test_df) == test_cfg["total_bars"], (
        f"Test bar count mismatch: {len(test_df)} vs {test_cfg['total_bars']}"
    )

    # Save to data/processed/
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train.csv"
    val_path = output_dir / "val.csv"
    test_path = output_dir / "test.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"  Saved Train: {train_path} ({len(train_df):,} bars, {train_df['session_date'].nunique()} days)")
    print(f"  Saved Val  : {val_path} ({len(val_df):,} bars, {val_df['session_date'].nunique()} days)")
    print(f"  Saved Test : {test_path} ({len(test_df):,} bars, {test_df['session_date'].nunique()} days)")

    return train_df, val_df, test_df


if __name__ == "__main__":
    partition_and_save_datasets()

