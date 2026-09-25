"""
Pairwise feature extraction module for candidate entity pairs.
Computes string similarity, token overlap, address/numeric compatibility,
and metadata signals using high-performance C++ rapidfuzz routines.
"""
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from rapidfuzz import fuzz, distance

from .normalize import (
    normalize_business_name,
    normalize_business_address,
)

FEATURE_NAMES = [
    # Name string similarities
    "name_ratio",
    "name_partial_ratio",
    "name_token_sort_ratio",
    "name_token_set_ratio",
    "name_jaro_winkler",
    "name_token_jaccard",
    "name_core_exact_match",
    "name_len_diff",
    "name_len_ratio",
    # Address string similarities
    "addr_missing",
    "addr_ratio",
    "addr_token_sort_ratio",
    "addr_token_set_ratio",
    "addr_partial_ratio",
    "addr_token_jaccard",
    "addr_number_jaccard",
    "addr_has_common_number",
    "addr_len_diff",
    # Metadata signals
    "country_exact_match",
    "target_is_s2",
    "target_is_s3",
    # Composite interaction features
    "name_x_addr_sort",
    "name_x_addr_num",
]


def jaccard_similarity(set1: set, set2: set) -> float:
    if not set1 or not set2:
        return 0.0
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return float(intersection) / float(union) if union > 0 else 0.0


def extract_pair_features(
    s1_record: Dict[str, Any],
    target_record: Dict[str, Any],
    target_id: str
) -> List[float]:
    """
    Extracts numerical feature vector for one (Source 1, Target) candidate pair.
    """
    # Name features
    s1_norm_name = s1_record.get("norm_name", "")
    s1_core_name = s1_record.get("core_name", "")
    s1_core_tokens = s1_record.get("core_tokens", [])
    
    t_norm_name = target_record.get("norm_name", "")
    t_core_name = target_record.get("core_name", "")
    t_core_tokens = target_record.get("core_tokens", [])
    
    name_ratio = fuzz.ratio(s1_norm_name, t_norm_name) / 100.0
    name_partial_ratio = fuzz.partial_ratio(s1_norm_name, t_norm_name) / 100.0
    name_sort_ratio = fuzz.token_sort_ratio(s1_norm_name, t_norm_name) / 100.0
    name_set_ratio = fuzz.token_set_ratio(s1_norm_name, t_norm_name) / 100.0
    name_jw = distance.JaroWinkler.similarity(s1_norm_name, t_norm_name)
    name_jaccard = jaccard_similarity(set(s1_core_tokens), set(t_core_tokens))
    name_core_exact = 1.0 if (s1_core_name and s1_core_name == t_core_name) else 0.0
    
    len1 = len(s1_norm_name)
    len2 = len(t_norm_name)
    name_len_diff = float(abs(len1 - len2))
    name_len_ratio = float(min(len1, len2)) / float(max(len1, len2, 1))
    
    # Address features
    s1_norm_addr = s1_record.get("norm_addr", "")
    s1_addr_nums = set(s1_record.get("addr_nums", []))
    
    t_norm_addr = target_record.get("norm_addr", "")
    t_addr_nums = set(target_record.get("addr_nums", []))
    
    addr_missing = 1.0 if (not s1_norm_addr or not t_norm_addr) else 0.0
    
    if addr_missing == 1.0:
        addr_ratio = 0.0
        addr_sort_ratio = 0.0
        addr_set_ratio = 0.0
        addr_partial_ratio = 0.0
        addr_jaccard = 0.0
        addr_num_jaccard = 0.0
        addr_has_common_num = 0.0
        addr_len_diff = 0.0
    else:
        addr_ratio = fuzz.ratio(s1_norm_addr, t_norm_addr) / 100.0
        addr_sort_ratio = fuzz.token_sort_ratio(s1_norm_addr, t_norm_addr) / 100.0
        addr_set_ratio = fuzz.token_set_ratio(s1_norm_addr, t_norm_addr) / 100.0
        addr_partial_ratio = fuzz.partial_ratio(s1_norm_addr, t_norm_addr) / 100.0
        addr_jaccard = jaccard_similarity(set(s1_norm_addr.split()), set(t_norm_addr.split()))
        addr_num_jaccard = jaccard_similarity(s1_addr_nums, t_addr_nums)
        common_nums = [n for n in s1_addr_nums.intersection(t_addr_nums) if len(n) >= 2]
        addr_has_common_num = 1.0 if common_nums else 0.0
        addr_len_diff = float(abs(len(s1_norm_addr) - len(t_norm_addr)))
        
    # Metadata features
    c1 = str(s1_record.get("country", "")).strip().upper()
    c2 = str(target_record.get("country", "")).strip().upper()
    country_exact = 1.0 if (c1 and c2 and c1 == c2) else 0.0
    
    is_s2 = 1.0 if target_id.startswith("S2-") else 0.0
    is_s3 = 1.0 if target_id.startswith("S3-") else 0.0
    
    # Interactions
    name_x_addr = name_sort_ratio * addr_sort_ratio
    name_x_num = name_sort_ratio * addr_has_common_num
    
    return [
        name_ratio,
        name_partial_ratio,
        name_sort_ratio,
        name_set_ratio,
        name_jw,
        name_jaccard,
        name_core_exact,
        name_len_diff,
        name_len_ratio,
        addr_missing,
        addr_ratio,
        addr_sort_ratio,
        addr_set_ratio,
        addr_partial_ratio,
        addr_jaccard,
        addr_num_jaccard,
        addr_has_common_num,
        addr_len_diff,
        country_exact,
        is_s2,
        is_s3,
        name_x_addr,
        name_x_num
    ]


def preprocess_record(name: str, addr: str, country: str) -> Dict[str, Any]:
    """
    Precomputes normalized fields once per record for fast repeated feature extraction.
    """
    norm_name, core_name, core_tokens = normalize_business_name(name)
    norm_addr, landmark, addr_nums = normalize_business_address(addr)
    return {
        "norm_name": norm_name,
        "core_name": core_name,
        "core_tokens": core_tokens,
        "norm_addr": norm_addr,
        "landmark": landmark,
        "addr_nums": addr_nums,
        "country": country
    }
