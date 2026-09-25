"""
End-to-End Pipeline Runner for the Business Entity Resolution Challenge.
Supports:
  --stage naive     : Runs baseline blocking + simple classifier, reports baseline metrics
  --stage optimized : Runs multi-key blocking + feature engineering + threshold tuning, reports metrics
  --test            : Generates candidate_pairs.tsv and matching_results.tsv for dataset/test
"""
import os
import sys
import time
import argparse
import subprocess
import numpy as np
import pandas as pd

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.load import load_source_tsv, load_ground_truth
from src.split import create_train_val_split
from src.normalize import preprocess_record
from src.block import InvertedIndexBlocker
from src.features import extract_pair_features, FEATURE_NAMES
from src.train import train_matching_model, tune_decision_threshold
from src.infer import run_full_test_inference
from src.evaluate import evaluate_predictions, evaluate_blocking


def run_pipeline(
    stage: str = "optimized",
    run_test: bool = False,
    val_size: int = 5000,
    train_size: int = 15000
):
    print("=" * 80)
    print(f"BUSINESS ENTITY RESOLUTION PIPELINE - STAGE: {stage.upper()}")
    print("=" * 80)
    t0_total = time.time()
    
    # ---------------------------------------------------------
    # 1. SPLIT & DATA PREPARATION
    # ---------------------------------------------------------
    print("\n[Step 1/5] Preparing Entity-Disjoint Train/Val Split...")
    t1 = time.time()
    gt_path = "dataset/train/train_ground_truth.tsv"
    s1_train_path = "dataset/train/train_source1.tsv"
    s2_train_path = "dataset/train/train_source2.tsv"
    s3_train_path = "dataset/train/train_source3.tsv"
    
    train_s1_ids, val_s1_ids = create_train_val_split(
        gt_path, val_size=val_size, train_size=train_size, random_seed=42
    )
    print(f"  Selected {len(train_s1_ids):,} training S1 entities and {len(val_s1_ids):,} validation S1 entities in {time.time()-t1:.2f}s.")
    
    # Load ground truth for evaluation
    full_gt = load_ground_truth(gt_path)
    val_gt = {s1: full_gt.get(s1, set()) for s1 in val_s1_ids}
    train_gt = {s1: full_gt.get(s1, set()) for s1 in train_s1_ids}
    
    # Collect all needed target IDs for training and validation
    all_needed_targets = set()
    for targets in val_gt.values():
        all_needed_targets.update(targets)
    for targets in train_gt.values():
        all_needed_targets.update(targets)
    print(f"  True target matches needed for train/val: {len(all_needed_targets):,}")
    
    # ---------------------------------------------------------
    # 2. LOAD & PREPROCESS RECORDS
    # ---------------------------------------------------------
    print("\n[Step 2/5] Loading and Normalizing Records...")
    t_load = time.time()
    df_s1 = load_source_tsv(s1_train_path)
    
    # Filter S1 to our split
    df_s1_val = df_s1[df_s1["entity_id"].isin(val_s1_ids)].copy()
    df_s1_train = df_s1[df_s1["entity_id"].isin(train_s1_ids)].copy()
    del df_s1
    
    # Preprocess S1 records
    s1_preprocessed = {}
    combined_s1 = pd.concat([df_s1_train, df_s1_val], ignore_index=True)
    for eid, name, addr, ctry in zip(
        combined_s1["entity_id"].values,
        combined_s1["business_name"].values,
        combined_s1["business_address"].values,
        combined_s1["country"].values
    ):
        s1_preprocessed[str(eid).strip()] = preprocess_record(str(name), str(addr), str(ctry))
    del combined_s1
        
    # Load S2 and S3 (initial batch + all needed true targets)
    print("  Loading target sources (S2 & S3)...")
    df_s2_sample = load_source_tsv(s2_train_path, nrows=120000)
    df_s3_sample = load_source_tsv(s3_train_path, nrows=120000)
    
    loaded_targets = set(df_s2_sample["entity_id"]).union(set(df_s3_sample["entity_id"]))
    missing_targets = all_needed_targets - loaded_targets
    print(f"  Initial target pool: {len(loaded_targets):,}. Missing true targets to fetch: {len(missing_targets):,}")
    
    if missing_targets:
        s2_missing_ids = {t for t in missing_targets if t.startswith("S2-")}
        s3_missing_ids = {t for t in missing_targets if t.startswith("S3-")}
        extra_dfs = []
        if s2_missing_ids:
            for chunk in pd.read_csv(s2_train_path, sep="\t", chunksize=250000, dtype=str):
                sub = chunk[chunk["entity_id"].isin(s2_missing_ids)]
                if not sub.empty:
                    extra_dfs.append(sub)
                    s2_missing_ids -= set(sub["entity_id"])
                    if not s2_missing_ids:
                        break
        if s3_missing_ids:
            for chunk in pd.read_csv(s3_train_path, sep="\t", chunksize=250000, dtype=str):
                sub = chunk[chunk["entity_id"].isin(s3_missing_ids)]
                if not sub.empty:
                    extra_dfs.append(sub)
                    s3_missing_ids -= set(sub["entity_id"])
                    if not s3_missing_ids:
                        break
        if extra_dfs:
            df_targets = pd.concat([df_s2_sample, df_s3_sample] + extra_dfs, ignore_index=True).drop_duplicates(subset=["entity_id"])
        else:
            df_targets = pd.concat([df_s2_sample, df_s3_sample], ignore_index=True).drop_duplicates(subset=["entity_id"])
    else:
        df_targets = pd.concat([df_s2_sample, df_s3_sample], ignore_index=True).drop_duplicates(subset=["entity_id"])
        
    print(f"  Total target records indexed: {len(df_targets):,} (loaded in {time.time()-t_load:.1f}s)")
    
    # Preprocess targets
    target_preprocessed = {}
    for eid, name, addr, ctry in zip(
        df_targets["entity_id"].values,
        df_targets["business_name"].values,
        df_targets["business_address"].values,
        df_targets["country"].values
    ):
        target_preprocessed[str(eid).strip()] = preprocess_record(str(name), str(addr), str(ctry))
        
    # ---------------------------------------------------------
    # 3. BLOCKING / CANDIDATE GENERATION
    # ---------------------------------------------------------
    print("\n[Step 3/5] Building Candidate Generation (Blocking) Index...")
    t_block = time.time()
    
    is_optimized = (stage == "optimized")
    max_freq = 500 if is_optimized else 2000
    max_cands = 20 if is_optimized else 5
    include_address = is_optimized
    
    blocker = InvertedIndexBlocker(max_key_frequency=max_freq, max_candidates_per_entity=max_cands)
    blocker.add_target_records(df_targets, include_address_keys=include_address)
    pruned = blocker.prune_high_frequency_keys()
    print(f"  Indexed {len(blocker.index):,} keys (pruned {pruned:,} high-frequency keys) in {time.time()-t_block:.2f}s.")
    
    # Generate candidates for validation set
    print("  Generating candidates for validation set...")
    t_cand = time.time()
    val_candidates = blocker.generate_candidates_for_s1(
        df_s1_val, include_address_keys=include_address, min_pre_score=20.0 if is_optimized else 10.0
    )
    print(f"  Generated val candidates in {time.time()-t_cand:.2f}s.")
    
    # Evaluate Blocking Metrics on Validation Split
    blocking_metrics = evaluate_blocking(val_gt, val_candidates, total_target_records=len(df_targets))
    print("\n" + "=" * 50)
    print(">>> BLOCKING METRICS (Validation Split) <<<")
    print(f"  Blocking Recall Ceiling : {blocking_metrics['blocking_recall_ceiling']*100:.2f}% ({blocking_metrics['covered_matches']}/{blocking_metrics['total_true_matches']} true matches)")
    print(f"  Reduction Ratio         : {blocking_metrics['reduction_ratio']*100:.6f}%")
    print(f"  Candidates Per Entity   : {blocking_metrics['candidates_per_entity']:.2f} (Total: {blocking_metrics['total_candidates']:,})")
    print("=" * 50)
    
    # Generate candidates for training set
    print("  Generating candidates for training set...")
    train_candidates = blocker.generate_candidates_for_s1(
        df_s1_train, include_address_keys=include_address, min_pre_score=20.0 if is_optimized else 10.0
    )
    
    # ---------------------------------------------------------
    # 4. PAIRWISE FEATURE EXTRACTION & MODEL TRAINING
    # ---------------------------------------------------------
    print("\n[Step 4/5] Extracting Features and Training Matching Model...")
    t_feat = time.time()
    
    train_pairs = []
    y_train = []
    
    for s1_id, cands in train_candidates.items():
        true_set = train_gt.get(s1_id, set())
        for cid in cands:
            if cid in target_preprocessed:
                label = 1 if cid in true_set else 0
                train_pairs.append((s1_id, cid))
                y_train.append(label)
        for true_id in true_set:
            if true_id in target_preprocessed and (s1_id, true_id) not in train_pairs:
                train_pairs.append((s1_id, true_id))
                y_train.append(1)
                
    print(f"  Constructed {len(train_pairs):,} training pairs: {sum(y_train):,} positives, {len(y_train)-sum(y_train):,} negatives.")
    
    # Feature extraction for training pairs
    X_train_list = []
    for s1_id, cid in train_pairs:
        f = extract_pair_features(s1_preprocessed[s1_id], target_preprocessed[cid], cid)
        X_train_list.append(f)
    X_train = np.array(X_train_list, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.int32)
    
    print("  Training LightGBM matching model...")
    n_est = 150 if is_optimized else 50
    lr = 0.05 if is_optimized else 0.1
    model = train_matching_model(X_train, y_train, FEATURE_NAMES, n_estimators=n_est, learning_rate=lr)
    
    # Feature extraction for validation pairs
    val_pairs = []
    for s1_id, cands in val_candidates.items():
        for cid in cands:
            if cid in target_preprocessed:
                val_pairs.append((s1_id, cid))
                
    X_val_list = []
    for s1_id, cid in val_pairs:
        f = extract_pair_features(s1_preprocessed[s1_id], target_preprocessed[cid], cid)
        X_val_list.append(f)
    X_val = np.array(X_val_list, dtype=np.float32) if X_val_list else np.empty((0, len(FEATURE_NAMES)))
    
    print("  Tuning decision threshold directly on macro-averaged F_0.5...")
    best_thresh, best_f05, best_metrics = tune_decision_threshold(
        model, val_pairs, X_val, val_gt
    )
    
    print("\n" + "=" * 50)
    print(">>> MATCHING MODEL METRICS (Validation Split) <<<")
    print(f"  Optimal Decision Threshold : {best_thresh:.2f}")
    print(f"  Validation Macro F_0.5      : {best_f05:.4f}")
    print(f"  Validation Macro Precision : {best_metrics.get('macro_precision', 0.0):.4f}")
    print(f"  Validation Macro Recall    : {best_metrics.get('macro_recall', 0.0):.4f}")
    print(f"  Total Validation Entities  : {best_metrics.get('num_entities', 0):,}")
    print("=" * 50)
    
    # ---------------------------------------------------------
    # 5. TEST SET INFERENCE
    # ---------------------------------------------------------
    if run_test:
        print("\n[Step 5/5] Running Full Test Set Inference...")
        run_full_test_inference(
            model=model,
            threshold=best_thresh,
            test_s1_path="dataset/test/test_source1.tsv",
            test_s2_path="dataset/test/test_source2.tsv",
            test_s3_path="dataset/test/test_source3.tsv",
            output_cand_path="output/candidate_pairs.tsv",
            output_match_path="output/matching_results.tsv",
            max_cands=max_cands,
            max_key_freq=max_freq,
            min_pre_score=20.0 if is_optimized else 10.0
        )
        
        # Run official challenge validator
        print("\n" + "=" * 60)
        print("RUNNING OFFICIAL CHALLENGE VALIDATOR")
        print("=" * 60)
        cmd = [
            sys.executable,
            "utils/validate_submission.py",
            "--matching", "output/matching_results.tsv",
            "--candidate", "output/candidate_pairs.tsv",
            "--test-dir", "dataset/test"
        ]
        ret = subprocess.run(cmd)
        if ret.returncode == 0:
            print("\n>>> SUBMISSION VALIDATION PASSED: PASS (Exit code 0) <<<")
        else:
            print("\n>>> SUBMISSION VALIDATION FAILED! <<<")
            
    print("\n" + "=" * 80)
    print(f"PIPELINE COMPLETED IN {time.time()-t0_total:.1f}s")
    print("=" * 80)
    return {
        "blocking_metrics": blocking_metrics,
        "best_threshold": best_thresh,
        "matching_metrics": best_metrics
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["naive", "optimized"], default="optimized", help="Pipeline stage to run")
    parser.add_argument("--test", action="store_true", help="Run full inference on dataset/test and generate outputs")
    parser.add_argument("--val-size", type=int, default=5000, help="Validation S1 size")
    parser.add_argument("--train-size", type=int, default=15000, help="Train S1 size")
    args = parser.parse_args()
    
    run_pipeline(stage=args.stage, run_test=args.test, val_size=args.val_size, train_size=args.train_size)
