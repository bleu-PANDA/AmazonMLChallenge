# Business Entity Resolution — ML Challenge 2026

End-to-end entity resolution pipeline that matches business records across three independent
data sources.  Given a deduplicated reference source (Source 1), the pipeline finds all
corresponding records in the noisy Source 2 and Source 3 files.

## Approach

| Stage | Method |
|---|---|
| **Text Normalization** | `text-unidecode` transliteration → lowercase → legal suffix expansion → punctuation stripping → landmark extraction |
| **Blocking / Candidate Generation** | Multi-key inverted index (token keys, 4-char prefix keys, sorted token-pair keys, address-number × word keys) with Counter-weighted candidate ranking |
| **Pairwise Features** | 23 features: name similarity (ratio, partial, token-sort, token-set, Jaro-Winkler, Jaccard), address similarity (same set + number overlap), country match, length diffs, interactions |
| **Matching Model** | LightGBM binary classifier (MIT License, ~10K parameters) |
| **Threshold Tuning** | Direct optimization on macro-averaged F_0.5 over entity-disjoint validation split |

## Model License

- **LightGBM** — MIT License — [github.com/microsoft/LightGBM](https://github.com/microsoft/LightGBM)
- Total model parameters: ~10K (gradient boosted tree with 150 estimators × 31 leaves)
- ✅ Well under the 8B parameter limit

## Reproduction — Step by Step

### 1. Environment setup

```bash
cd code/business_entity_resolution
pip install -r requirements.txt
```

### 2. Run the full pipeline (validation + test inference + validation check)

```bash
# From the student_resource/ root directory:
python3 -u code/business_entity_resolution/run_pipeline.py \
    --stage optimized --test \
    --val-size 5000 --train-size 15000
```

This single command:
1. Creates an entity-disjoint train/val split from `dataset/train/`
2. Builds the blocking index and measures **blocking recall ceiling**
3. Extracts pairwise features and trains the LightGBM classifier
4. Tunes the decision threshold directly on macro-averaged F_0.5
5. Runs inference on `dataset/test/` (country-by-country streaming)
6. Writes `output/candidate_pairs.tsv` and `output/matching_results.tsv`
7. Runs the official `utils/validate_submission.py` validator

### 3. Validate outputs manually (optional)

```bash
python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

## Directory Layout

```
code/business_entity_resolution/
├── run_pipeline.py          # Main entry point (--stage, --test, --val-size, --train-size)
├── requirements.txt         # Pinned dependencies
├── README.md                # This file
└── src/
    ├── __init__.py
    ├── load.py              # TSV loading utilities (sep="\t", dtype=str)
    ├── normalize.py         # Text normalization (names, addresses, landmarks)
    ├── block.py             # Inverted index blocking + candidate ranking
    ├── features.py          # 23-feature pairwise extraction (rapidfuzz-based)
    ├── train.py             # LightGBM training + F_0.5 threshold tuning
    ├── infer.py             # Country-streaming test inference
    ├── evaluate.py          # Macro F_0.5, blocking metrics
    └── split.py             # Entity-disjoint train/val split generator
```

## Key Design Decisions

1. **Country as partition, not filter**: The pipeline dynamically partitions by country
   during both blocking and inference.  No country values are hardcoded; France (unseen in
   training) is handled identically to US and India.

2. **text-unidecode for transliteration**: Non-Latin scripts (Devanagari, Tamil, Kannada,
   French accents) are transliterated to ASCII before tokenization.  This lets the same
   blocking keys match across script variants (e.g. "एसएस" → "eses" ≈ "ss").

3. **Counter-weighted candidate ranking**: When blocking retrieves > K candidates, they are
   ranked by `key_overlap_count × 15 + name_similarity + 0.2 × addr_similarity` and the
   top-K are kept.  This significantly boosts blocking recall vs. naive top-K by similarity.

4. **Lazy target preprocessing**: During test inference, only targets that actually appear
   as candidates are fully preprocessed (normalized name/address/tokens).  This avoids
   preprocessing millions of targets that would never be scored.

5. **No external data**: The entire pipeline uses only the provided train/test TSV files.
   No geocoding APIs, business registries, or internet lookups are used.

## Contributors

- **bleu-PANDA** ([@bleu-PANDA](https://github.com/bleu-PANDA))

