"""
Model Training and Threshold Tuning Module.
Trains a gradient boosted decision tree (LightGBM) on pairwise features.
Optimizes decision threshold directly for macro-averaged F_0.5 on the validation split.
"""
from typing import Dict, List, Set, Tuple, Optional
import numpy as np
import lightgbm as lgb

from .evaluate import evaluate_predictions


def train_matching_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: List[str],
    num_leaves: int = 31,
    learning_rate: float = 0.05,
    n_estimators: int = 150,
    random_state: int = 42
) -> lgb.LGBMClassifier:
    """
    Trains a LightGBM classifier (MIT License, << 8B parameters) for entity matching.
    """
    # Scale positive weight if class imbalance is significant
    pos_count = int(np.sum(y_train == 1))
    neg_count = int(np.sum(y_train == 0))
    scale_pos_weight = 1.0  # Keep balanced to preserve precision for F_0.5
    
    model = lgb.LGBMClassifier(
        num_leaves=num_leaves,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
        n_jobs=-1,
        verbose=-1
    )
    model.fit(X_train, y_train, feature_name=feature_names)
    return model


def tune_decision_threshold(
    model: lgb.LGBMClassifier,
    val_pairs: List[Tuple[str, str]],
    X_val: np.ndarray,
    val_ground_truth: Dict[str, Set[str]],
    thresholds: Optional[List[float]] = None
) -> Tuple[float, float, Dict[str, float]]:
    """
    Evaluates candidate thresholds directly on macro-averaged F_0.5 (including singletons)
    and returns (best_threshold, best_f05, best_metrics).
    """
    if thresholds is None:
        thresholds = [round(t, 2) for t in np.arange(0.20, 0.95, 0.05)]
        
    probabilities = model.predict_proba(X_val)[:, 1]
    
    # Group probabilities by s1_entity_id
    s1_to_scored_cands: Dict[str, List[Tuple[float, str]]] = {
        s1: [] for s1 in val_ground_truth.keys()
    }
    for (s1_id, cand_id), prob in zip(val_pairs, probabilities):
        if s1_id in s1_to_scored_cands:
            s1_to_scored_cands[s1_id].append((float(prob), cand_id))
            
    best_threshold = 0.5
    best_f05 = -1.0
    best_metrics = {}
    
    for thresh in thresholds:
        preds = {}
        for s1_id, scored_list in s1_to_scored_cands.items():
            matched = {cid for p, cid in scored_list if p >= thresh}
            preds[s1_id] = matched
            
        metrics = evaluate_predictions(val_ground_truth, preds)
        score = metrics["macro_f05"]
        if score > best_f05:
            best_f05 = score
            best_threshold = thresh
            best_metrics = metrics
            
    return best_threshold, best_f05, best_metrics
