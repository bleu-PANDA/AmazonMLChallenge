"""
Data loading utilities for TSV source files.
Explicitly uses sep='\\t' and handles missing fields gracefully.
"""
import os
from typing import Optional, Dict, Set
import pandas as pd


def load_source_tsv(
    filepath: str,
    nrows: Optional[int] = None,
    usecols: Optional[list] = None
) -> pd.DataFrame:
    """
    Loads a source TSV file (source1, source2, or source3).
    Ensures tab delimiter is used and missing text fields are treated as empty strings.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
        
    df = pd.read_csv(
        filepath,
        sep="\t",
        nrows=nrows,
        usecols=usecols,
        dtype=str,
        keep_default_na=False
    )
    for col in ["entity_id", "business_name", "business_address", "country"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    return df


def load_ground_truth(
    filepath: str,
    nrows: Optional[int] = None
) -> Dict[str, Set[str]]:
    """
    Loads train_ground_truth.tsv into a mapping:
    source1_entity_id -> set of matched entity_ids
    Uses fast zip iteration.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Ground truth file not found: {filepath}")
        
    df = pd.read_csv(
        filepath,
        sep="\t",
        nrows=nrows,
        dtype=str,
        keep_default_na=False
    )
    
    gt_mapping: Dict[str, Set[str]] = {}
    s1_vals = df["source1_entity_id"].values
    matched_vals = df["matched_entity_ids"].values
    
    for s1, m in zip(s1_vals, matched_vals):
        s1_id = str(s1).strip()
        matched_str = str(m).strip()
        if matched_str and matched_str != "nan":
            matched_ids = {mid.strip() for mid in matched_str.split(",") if mid.strip()}
        else:
            matched_ids = set()
        gt_mapping[s1_id] = matched_ids
        
    return gt_mapping
