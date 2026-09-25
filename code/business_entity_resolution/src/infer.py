"""
Inference module for producing final submission TSV files.
Optimized for large-scale test sets (~10M target records):
- Streams targets by country from disk (avoids loading all 10M at once)
- Only preprocesses targets that appear as candidates (lazy preprocess)
- Processes S1 entities in chunks for flat memory

Strictly enforces:
- Exactly one row per Source 1 test entity in identical order as test_source1.tsv
- S2/S3 IDs only, no duplicates, matched ⊆ candidates
- Tab-separated output, no quoting
"""
import gc
import os
import time
from typing import Dict, Set, List
import pandas as pd
import numpy as np
import lightgbm as lgb

from .normalize import preprocess_record
from .block import InvertedIndexBlocker
from .features import extract_pair_features


def _load_targets_for_country(s2_path: str, s3_path: str, country: str) -> pd.DataFrame:
    """Streams S2+S3 TSVs and returns only records matching *country*."""
    frames = []
    for path in (s2_path, s3_path):
        for chunk in pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False,
                                 chunksize=500_000):
            sub = chunk[chunk["country"].str.strip() == country]
            if not sub.empty:
                frames.append(sub)
    if not frames:
        return pd.DataFrame(columns=["entity_id", "business_name", "business_address", "country"])
    df = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["entity_id"])
    for col in df.columns:
        df[col] = df[col].astype(str).str.strip()
    return df


def run_full_test_inference(
    model: lgb.LGBMClassifier,
    threshold: float,
    test_s1_path: str = "dataset/test/test_source1.tsv",
    test_s2_path: str = "dataset/test/test_source2.tsv",
    test_s3_path: str = "dataset/test/test_source3.tsv",
    output_cand_path: str = "output/candidate_pairs.tsv",
    output_match_path: str = "output/matching_results.tsv",
    max_cands: int = 20,
    max_key_freq: int = 500,
    min_pre_score: float = 20.0
):
    """
    Executes test inference partitioned by country.
    Targets are loaded per-country (not all at once) to stay under 4 GB RAM.
    """
    t0 = time.time()
    os.makedirs(os.path.dirname(output_cand_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(output_match_path) or ".", exist_ok=True)

    print("\n" + "=" * 60)
    print("STARTING TEST SET INFERENCE")
    print(f"  Model Decision Threshold : {threshold:.2f}")
    print(f"  Candidate Cap per Entity : {max_cands}")
    print("=" * 60)

    # ---- Load test S1 (preserves row order for output) ----
    print("Loading test_source1.tsv ...")
    df_s1 = pd.read_csv(test_s1_path, sep="\t", dtype=str, keep_default_na=False)
    for col in df_s1.columns:
        df_s1[col] = df_s1[col].astype(str).str.strip()
    s1_all_ids = list(df_s1["entity_id"].values)
    print(f"  Total test S1 entities: {len(s1_all_ids):,}")

    results_candidates: Dict[str, str] = {}
    results_matches: Dict[str, str] = {}

    countries = list(df_s1["country"].unique())
    print(f"  Countries: {countries}")

    for ctry in countries:
        t_c = time.time()
        print(f"\n>>> Processing Country: {ctry} <<<")
        sub_s1 = df_s1[df_s1["country"] == ctry].copy()
        print(f"  S1 entities: {len(sub_s1):,}")

        # ---- Stream targets for this country only ----
        print(f"  Loading targets for {ctry} (streaming) ...")
        t_load = time.time()
        sub_targets = _load_targets_for_country(test_s2_path, test_s3_path, ctry)
        print(f"  Target records: {len(sub_targets):,} (loaded in {time.time()-t_load:.1f}s)")

        # ---- Build blocking index ----
        print("  Building blocking index ...")
        t_idx = time.time()
        blocker = InvertedIndexBlocker(
            max_key_frequency=max_key_freq,
            max_candidates_per_entity=max_cands
        )
        blocker.add_target_records(sub_targets, include_address_keys=True)
        pruned = blocker.prune_high_frequency_keys()
        print(f"  Index: {len(blocker.index):,} keys, pruned {pruned:,} "
              f"({time.time()-t_idx:.1f}s)")

        # ---- Generate candidates for every S1 in this country ----
        print("  Generating candidates ...")
        t_cand = time.time()
        all_cands = blocker.generate_candidates_for_s1(
            sub_s1, include_address_keys=True, min_pre_score=min_pre_score
        )
        print(f"  Candidates done ({time.time()-t_cand:.1f}s)")

        # ---- Lazy-preprocess only the targets that appear as candidates ----
        needed_target_ids = set()
        for cset in all_cands.values():
            needed_target_ids.update(cset)
        print(f"  Unique candidate targets to preprocess: {len(needed_target_ids):,}")

        # Build a quick lookup from target df
        target_idx = sub_targets.set_index("entity_id")
        target_prepped: Dict[str, dict] = {}
        for tid in needed_target_ids:
            if tid in target_idx.index:
                row = target_idx.loc[tid]
                target_prepped[tid] = preprocess_record(
                    str(row["business_name"]),
                    str(row["business_address"]),
                    str(row["country"])
                )
        del target_idx
        print(f"  Preprocessed {len(target_prepped):,} target records")

        # ---- Score candidate pairs in S1 chunks ----
        chunk_size = 50_000
        s1_ids_arr = sub_s1["entity_id"].values
        s1_names_arr = sub_s1["business_name"].values
        s1_addrs_arr = sub_s1["business_address"].values
        s1_ctrys_arr = sub_s1["country"].values

        for start in range(0, len(sub_s1), chunk_size):
            end = min(start + chunk_size, len(sub_s1))
            # Preprocess S1 chunk
            s1_prepped: Dict[str, dict] = {}
            for j in range(start, end):
                eid = str(s1_ids_arr[j])
                s1_prepped[eid] = preprocess_record(
                    str(s1_names_arr[j]),
                    str(s1_addrs_arr[j]),
                    str(s1_ctrys_arr[j])
                )

            # Collect pairs for this chunk
            pairs: List[tuple] = []
            for j in range(start, end):
                s1_id = str(s1_ids_arr[j])
                for cid in all_cands.get(s1_id, set()):
                    if cid in target_prepped:
                        pairs.append((s1_id, cid))

            # Feature extraction + scoring
            matched_dict: Dict[str, List[str]] = {}
            if pairs:
                feats = []
                for s1_id, cid in pairs:
                    feats.append(extract_pair_features(
                        s1_prepped[s1_id], target_prepped[cid], cid
                    ))
                X = np.array(feats, dtype=np.float32)
                probs = model.predict_proba(X)[:, 1]
                for (s1_id, cid), prob in zip(pairs, probs):
                    if prob >= threshold:
                        matched_dict.setdefault(s1_id, []).append(cid)

            # Store formatted results
            for j in range(start, end):
                s1_id = str(s1_ids_arr[j])
                cands = all_cands.get(s1_id, set())
                results_candidates[s1_id] = ",".join(sorted(cands))
                m_list = matched_dict.get(s1_id, [])
                valid = sorted(set(m_list).intersection(cands))
                results_matches[s1_id] = ",".join(valid)

            print(f"    Scored {end:,} / {len(sub_s1):,} S1 "
                  f"({end/len(sub_s1)*100:.0f}%)")

        del blocker, sub_targets, target_prepped, all_cands, sub_s1
        gc.collect()
        print(f"  Country {ctry} done in {time.time()-t_c:.1f}s")

    # ---- Write output TSVs ----
    print("\nWriting output TSVs ...")
    with open(output_cand_path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1_id in s1_all_ids:
            f.write(f"{s1_id}\t{results_candidates.get(s1_id, '')}\n")

    with open(output_match_path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s1_id in s1_all_ids:
            f.write(f"{s1_id}\t{results_matches.get(s1_id, '')}\n")

    print(f"  {len(s1_all_ids):,} rows written to each file")
    print(f"TEST INFERENCE COMPLETED IN {time.time()-t0:.1f}s")
    print("=" * 60)
