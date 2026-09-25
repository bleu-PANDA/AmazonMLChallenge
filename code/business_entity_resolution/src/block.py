"""
Blocking and Candidate Generation Module.
Provides inverted index candidate generation across S2 and S3 records.
Implements:
1. Multi-key token, prefix, and address-based blocking
2. Frequency pruning for hyper-frequent keys to avoid candidate explosions
3. Key-overlap weighted candidate ranking to select top-K candidates per Source 1 entity
"""
from collections import defaultdict, Counter
from typing import Dict, List, Set, Tuple
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
import re
import text_unidecode

from .normalize import (
    normalize_business_name,
    normalize_business_address,
    LEGAL_SUFFIXES_SET,
)

GENERIC_STOPWORDS = LEGAL_SUFFIXES_SET.union({
    "and", "the", "of", "in", "for", "to", "at", "by", "on", "with", "from",
    "services", "enterprises", "solutions", "international", "group", "holdings",
    "trading", "consultants", "india", "usa", "france"
})

# Fast inlined key generation for bulk indexing (avoids full normalize overhead)
_LEGAL_EXPANSIONS_FAST = {
    "corp": "corporation", "corporation": "corporation",
    "inc": "incorporated", "incorporated": "incorporated",
    "ltd": "limited", "limited": "limited",
    "pvt": "private", "private": "private",
    "co": "company", "company": "company",
    "llc": "llc", "llp": "llp",
    "sarl": "sarl", "sas": "sas", "sa": "sa", "sci": "sci",
    "sasu": "sasu", "snc": "snc", "eurl": "eurl",
}
_PUNCT_RE = re.compile(r"[^\w\s]")
_URL_RE = re.compile(r"https?://\S+|www\.\S+|\.(?:com|org|net|in|fr|io|co)\b", re.I)


def _fast_name_tokens(name: str) -> list:
    """Ultra-fast name token extraction for bulk indexing — skips full normalize."""
    if not name or name == "nan":
        return []
    t = text_unidecode.unidecode(name).lower()
    t = _URL_RE.sub(" ", t)
    t = t.replace("&", " and ")
    t = _PUNCT_RE.sub(" ", t)
    tokens = [_LEGAL_EXPANSIONS_FAST.get(w, w) for w in t.split() if w]
    return [
        w for w in tokens
        if len(w) >= 3 and w not in GENERIC_STOPWORDS and w not in LEGAL_SUFFIXES_SET
    ]


def _fast_addr_nums(addr: str) -> list:
    """Fast numeric token extraction from address."""
    if not addr or addr == "nan":
        return []
    t = text_unidecode.unidecode(addr).lower()
    nums = re.findall(r"\b\d+\b", t)
    return [n.lstrip("0") for n in nums if len(n.lstrip("0")) >= 2]


def _fast_addr_words(addr: str) -> list:
    """Fast meaningful word extraction from address."""
    if not addr or addr == "nan":
        return []
    t = text_unidecode.unidecode(addr).lower()
    t = _PUNCT_RE.sub(" ", t)
    return [
        w for w in t.split()
        if len(w) >= 4 and w not in GENERIC_STOPWORDS and w not in {
            "road", "street", "avenue", "lane", "drive", "court",
            "boulevard", "floor", "unit", "apartment", "block", "plot",
            "null", "near", "behind"
        }
    ]


def generate_blocking_keys(
    name: str,
    address: str,
    country: str,
    include_address_keys: bool = True
) -> Set[str]:
    """
    Generates blocking keys for a record.
    country is prefixed to every key to ensure country-partitioned candidate lookup.
    """
    keys = set()
    norm_country = str(country).strip().upper() if country else "UNKNOWN"

    meaningful_tokens = _fast_name_tokens(name)

    for tok in meaningful_tokens[:4]:
        keys.add(f"{norm_country}:tok:{tok}")
        if len(tok) >= 4:
            keys.add(f"{norm_country}:p4:{tok[:4]}")

    if len(meaningful_tokens) >= 2:
        pair_key = "_".join(sorted(meaningful_tokens[:2]))
        keys.add(f"{norm_country}:pair:{pair_key}")

    if include_address_keys and address:
        num_tokens = _fast_addr_nums(address)
        addr_words = _fast_addr_words(address)

        for num in num_tokens[:2]:
            for aw in addr_words[:2]:
                keys.add(f"{norm_country}:num_addr:{num}_{aw}")
            if meaningful_tokens:
                keys.add(f"{norm_country}:num_name:{num}_{meaningful_tokens[0]}")

    return keys


class InvertedIndexBlocker:
    """
    Inverted index for candidate generation across multiple sources.
    Uses Counter-based key overlap weighting for candidate retrieval.
    """
    def __init__(self, max_key_frequency: int = 500, max_candidates_per_entity: int = 20):
        self.max_key_frequency = max_key_frequency
        self.max_candidates_per_entity = max_candidates_per_entity
        self.index: Dict[str, List[str]] = defaultdict(list)
        self.target_names: Dict[str, str] = {}   # id -> raw name (for scoring)
        self.target_addrs: Dict[str, str] = {}   # id -> raw addr (for scoring)

    def add_target_records(self, df_targets: pd.DataFrame, include_address_keys: bool = True):
        """Bulk-indexes target records with fast inlined key generation."""
        eids = df_targets["entity_id"].values
        names = df_targets["business_name"].values
        addrs = df_targets["business_address"].values
        ctrys = df_targets["country"].values

        for i in range(len(eids)):
            eid = str(eids[i]).strip()
            name = str(names[i])
            addr = str(addrs[i])
            country = str(ctrys[i])
            self.target_names[eid] = name
            self.target_addrs[eid] = addr

            keys = generate_blocking_keys(name, addr, country, include_address_keys)
            for k in keys:
                self.index[k].append(eid)

            if (i + 1) % 500000 == 0:
                print(f"    ... indexed {i+1:,} / {len(eids):,} targets")

    def prune_high_frequency_keys(self) -> int:
        pruned_count = 0
        keys_to_remove = [k for k, v in self.index.items() if len(v) > self.max_key_frequency]
        for k in keys_to_remove:
            del self.index[k]
        return len(keys_to_remove)

    def generate_candidates_for_s1(
        self,
        df_s1: pd.DataFrame,
        include_address_keys: bool = True,
        min_pre_score: float = 20.0
    ) -> Dict[str, Set[str]]:
        candidates: Dict[str, Set[str]] = {}

        eids = df_s1["entity_id"].values
        names = df_s1["business_name"].values
        addrs = df_s1["business_address"].values
        ctrys = df_s1["country"].values

        for i in range(len(eids)):
            s1_id = str(eids[i]).strip()
            name1 = str(names[i])
            addr1 = str(addrs[i])
            country1 = str(ctrys[i])

            s1_keys = generate_blocking_keys(name1, addr1, country1, include_address_keys)

            cand_counts = Counter()
            for k in s1_keys:
                if k in self.index:
                    cand_counts.update(self.index[k])

            if not cand_counts:
                candidates[s1_id] = set()
                continue

            if len(cand_counts) <= self.max_candidates_per_entity:
                candidates[s1_id] = set(cand_counts.keys())
            else:
                scored = []
                for cid, key_matches in cand_counts.items():
                    cname = self.target_names.get(cid, "")
                    caddr = self.target_addrs.get(cid, "")
                    sim = fuzz.token_sort_ratio(name1, cname)
                    score = key_matches * 15.0 + sim
                    if addr1 and caddr and addr1 != "nan" and caddr != "nan":
                        addr_sim = fuzz.token_set_ratio(addr1, caddr)
                        score += 0.2 * addr_sim
                    scored.append((score, cid))

                scored.sort(key=lambda x: x[0], reverse=True)
                selected = {
                    cid for score, cid in scored[:self.max_candidates_per_entity]
                    if score >= min_pre_score
                }
                candidates[s1_id] = selected

            if (i + 1) % 100000 == 0:
                print(f"    ... queried {i+1:,} / {len(eids):,} S1 entities")

        return candidates
