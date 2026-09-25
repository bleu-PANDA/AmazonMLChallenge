"""
Text normalization module for business entity names and addresses.
Implements rule-based, generic normalization that generalizes across US, India, France,
and any other country without hardcoding per-country filters.
"""
import re
from typing import Tuple, List, Set
import text_unidecode

# Generic legal term dictionary mapping variants to canonical forms
LEGAL_EXPANSIONS = {
    # US / UK / India / Common English
    "corp": "corporation",
    "corporation": "corporation",
    "inc": "incorporated",
    "incorporated": "incorporated",
    "ltd": "limited",
    "limited": "limited",
    "pvt": "private",
    "private": "private",
    "co": "company",
    "company": "company",
    "llc": "llc",
    "llp": "llp",
    "pllc": "pllc",
    "pc": "pc",
    "assoc": "associates",
    "associates": "associates",
    "intl": "international",
    "international": "international",
    "grp": "group",
    "group": "group",
    "tech": "technology",
    "technologies": "technology",
    "serv": "services",
    "services": "services",
    "ent": "enterprises",
    "enterprises": "enterprises",
    # French legal forms
    "sarl": "sarl",
    "sas": "sas",
    "sasu": "sasu",
    "sa": "sa",
    "sci": "sci",
    "snc": "snc",
    "eurl": "eurl",
    "ets": "etablissements",
    "etablissements": "etablissements",
    "ste": "societe",
    "societe": "societe",
    "cie": "company",
}

# Legal suffixes to identify and separate from core business identity
LEGAL_SUFFIXES_SET = {
    "corporation", "corp", "incorporated", "inc", "limited", "ltd",
    "private", "pvt", "company", "co", "llc", "llp", "pllc", "pc",
    "sarl", "sas", "sasu", "sa", "sci", "snc", "eurl", "etablissements", "societe"
}

# Road and address component abbreviations
ROAD_EXPANSIONS = {
    "rd": "road",
    "st": "street",
    "saint": "street",
    "ave": "avenue",
    "av": "avenue",
    "blvd": "boulevard",
    "bd": "boulevard",
    "ln": "lane",
    "dr": "drive",
    "ct": "court",
    "pkwy": "parkway",
    "hwy": "highway",
    "ter": "terrace",
    "terr": "terrace",
    "cir": "circle",
    "pl": "place",
    "sq": "square",
    "apt": "apartment",
    "ste": "suite",
    "fl": "floor",
    # French road types
    "r": "rue",
    "all": "allee",
    "imp": "impasse",
    "chem": "chemin",
    "rte": "route",
}

LANDMARK_PATTERN = re.compile(
    r"\b(near|behind|opp|opposite|next to|beside|in front of|close to|adjacent to|pres de|proche de|face a)\s+([^,;]+)",
    flags=re.IGNORECASE
)

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+|\.(?:com|org|net|in|fr|io|co)\b", flags=re.IGNORECASE)
PUNCT_PATTERN = re.compile(r"[^\w\s]")
MULTIPLE_SPACES = re.compile(r"\s+")
CONSECUTIVE_REPEATS = re.compile(r"([a-z])\1+")


def normalize_business_name(raw_name: str) -> Tuple[str, str, List[str]]:
    """
    Normalizes a business name:
    - Transliterates non-Latin scripts (Devanagari, Tamil, etc.) and strips accents
    - Removes web URLs / domain extensions / dba prefix
    - Replaces '&' with 'and', '+' with 'plus'
    - Lowercases, expands legal abbreviations
    - Strips punctuation and excessive whitespace
    
    Returns:
      (normalized_full_name, core_name, core_tokens)
    """
    if not isinstance(raw_name, str) or not raw_name.strip():
        return "", "", []
        
    # Transliterate to ASCII
    text = text_unidecode.unidecode(raw_name).lower()
    
    # Strip URL/domain noise
    text = URL_PATTERN.sub(" ", text)
    
    # Replace symbols
    text = re.sub(r"&", " and ", text)
    text = re.sub(r"\+", " plus ", text)
    text = re.sub(r"\b(dba)\b", " ", text)
    
    # Remove punctuation
    text = PUNCT_PATTERN.sub(" ", text)
    
    raw_tokens = text.split()
    expanded_tokens = [LEGAL_EXPANSIONS.get(t, t) for t in raw_tokens if t]
    
    # Core tokens: exclude legal suffixes and common generic filler words
    core_tokens = [
        t for t in expanded_tokens
        if t not in LEGAL_SUFFIXES_SET and t not in {"and", "the", "of", "in"}
    ]
    
    full_norm = " ".join(expanded_tokens)
    core_norm = " ".join(core_tokens)
    
    return full_norm, core_norm, core_tokens


def normalize_business_address(raw_address: str) -> Tuple[str, str, List[str]]:
    """
    Normalizes a business address:
    - Transliterates non-Latin scripts and strips accents
    - Extracts landmark phrases (e.g. 'near X') into a separate feature
    - Expands road abbreviations (rd -> road, st -> street, etc.)
    - Removes 'null', 'nan'
    - Extracts numbers (house numbers, postal codes)
    
    Returns:
      (normalized_address, landmark, numeric_tokens)
    """
    if not isinstance(raw_address, str) or not raw_address.strip() or raw_address.lower() in ("null", "nan"):
        return "", "", []
        
    text = text_unidecode.unidecode(raw_address).lower()
    
    # Extract landmark if present
    landmark = ""
    lm_match = LANDMARK_PATTERN.search(text)
    if lm_match:
        landmark = lm_match.group(0).strip()
        # Remove landmark from core address string to reduce noise
        text = text[:lm_match.start()] + " " + text[lm_match.end():]
        
    # Replace symbols
    text = PUNCT_PATTERN.sub(" ", text)
    
    # Extract numbers (house numbers, pins, postal codes)
    raw_nums = re.findall(r"\b\d+\b", text)
    # Strip leading zeros for robust matching (e.g. 0684 -> 684)
    numeric_tokens = [n.lstrip("0") for n in raw_nums if len(n.lstrip("0")) >= 1]
    
    words = text.split()
    expanded_words = [
        ROAD_EXPANSIONS.get(w, w) for w in words
        if w not in {"null", "nan"}
    ]
    
    norm_address = " ".join(expanded_words)
    return norm_address, landmark, numeric_tokens


def preprocess_record(name: str, addr: str, country: str) -> dict:
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

