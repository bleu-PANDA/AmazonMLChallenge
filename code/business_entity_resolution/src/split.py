"""
Validation split generator.
Extracts an entity-disjoint validation set and training set from train_source1 and train_ground_truth.
Ensures zero leakage across Source 1 entities and preserves country/singleton distributions.
"""
import random
from typing import Set, Tuple
import pandas as pd


def create_train_val_split(
    ground_truth_path: str,
    val_size: int = 10000,
    train_size: int = 30000,
    random_seed: int = 42
) -> Tuple[Set[str], Set[str]]:
    """
    Selects disjoint sets of source1_entity_ids for validation and training.
    Uses fast column iteration instead of slow iterrows().
    """
    random.seed(random_seed)
    
    df_gt = pd.read_csv(ground_truth_path, sep="\t", dtype=str, keep_default_na=False)
    
    s1_ids = df_gt["source1_entity_id"].values
    matched_col = df_gt["matched_entity_ids"].values
    
    singletons = []
    multi_matches = []
    
    for s1_id, matched in zip(s1_ids, matched_col):
        m = str(matched).strip()
        if not m or m == "nan":
            singletons.append(str(s1_id).strip())
        else:
            multi_matches.append(str(s1_id).strip())
            
    random.shuffle(singletons)
    random.shuffle(multi_matches)
    
    total = len(s1_ids)
    val_singleton_count = int(val_size * (len(singletons) / total))
    val_multi_count = val_size - val_singleton_count
    
    train_singleton_count = int(train_size * (len(singletons) / total))
    train_multi_count = train_size - train_singleton_count
    
    val_ids = set(singletons[:val_singleton_count] + multi_matches[:val_multi_count])
    train_ids = set(
        singletons[val_singleton_count:val_singleton_count + train_singleton_count] +
        multi_matches[val_multi_count:val_multi_count + train_multi_count]
    )
    
    assert len(val_ids.intersection(train_ids)) == 0, "Train and Val S1 entities must be disjoint!"
    return train_ids, val_ids
