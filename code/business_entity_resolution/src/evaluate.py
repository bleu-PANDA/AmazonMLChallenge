"""
Evaluation metrics for Entity Resolution.
Implements Macro-averaged F_0.5, Precision, Recall, Blocking Recall Ceiling, and Reduction Ratio.
"""
from typing import Dict, Set, List, Tuple
import numpy as np


def compute_entity_metrics(true_matches: Set[str], pred_matches: Set[str]) -> Tuple[float, float, float]:
    """
    Computes (precision, recall, f05) for a single Source 1 entity.
    Singletons:
      - true empty, pred empty: precision=1.0, recall=1.0, f05=1.0
      - true empty, pred non-empty: precision=0.0, recall=0.0, f05=0.0
    Non-singletons:
      - true non-empty, pred empty: precision=0.0, recall=0.0, f05=0.0
      - true non-empty, pred non-empty:
          p = tp / len(pred)
          r = tp / len(true)
          f05 = (1.25 * p * r) / (0.25 * p + r) if tp > 0 else 0.0
    """
    if len(true_matches) == 0:
        if len(pred_matches) == 0:
            return 1.0, 1.0, 1.0
        else:
            return 0.0, 0.0, 0.0
            
    if len(pred_matches) == 0:
        return 0.0, 0.0, 0.0
        
    tp = len(true_matches.intersection(pred_matches))
    if tp == 0:
        return 0.0, 0.0, 0.0
        
    p = tp / len(pred_matches)
    r = tp / len(true_matches)
    f05 = (1.25 * p * r) / (0.25 * p + r)
    return p, r, f05


def evaluate_predictions(
    ground_truth: Dict[str, Set[str]],
    predictions: Dict[str, Set[str]]
) -> Dict[str, float]:
    """
    Calculates macro-averaged F_0.5, Precision, and Recall across all S1 entities.
    """
    precisions = []
    recalls = []
    f05s = []
    
    for s1_id, true_set in ground_truth.items():
        pred_set = predictions.get(s1_id, set())
        p, r, f05 = compute_entity_metrics(true_set, pred_set)
        precisions.append(p)
        recalls.append(r)
        f05s.append(f05)
        
    return {
        "macro_f05": float(np.mean(f05s)),
        "macro_precision": float(np.mean(precisions)),
        "macro_recall": float(np.mean(recalls)),
        "num_entities": len(ground_truth)
    }


def evaluate_blocking(
    ground_truth: Dict[str, Set[str]],
    candidate_pairs: Dict[str, Set[str]],
    total_target_records: int
) -> Dict[str, float]:
    """
    Computes blocking recall ceiling and reduction ratio.
    """
    total_true_matches = 0
    covered_matches = 0
    total_candidates_generated = 0
    
    num_s1 = len(ground_truth)
    for s1_id, true_set in ground_truth.items():
        total_true_matches += len(true_set)
        candidates = candidate_pairs.get(s1_id, set())
        total_candidates_generated += len(candidates)
        covered_matches += len(true_set.intersection(candidates))
        
    recall_ceiling = (covered_matches / total_true_matches) if total_true_matches > 0 else 1.0
    total_possible_pairs = num_s1 * total_target_records
    reduction_ratio = (
        1.0 - (total_candidates_generated / total_possible_pairs)
    ) if total_possible_pairs > 0 else 1.0
    
    return {
        "blocking_recall_ceiling": recall_ceiling,
        "reduction_ratio": reduction_ratio,
        "total_true_matches": total_true_matches,
        "covered_matches": covered_matches,
        "candidates_per_entity": total_candidates_generated / num_s1 if num_s1 > 0 else 0.0,
        "total_candidates": total_candidates_generated
    }
