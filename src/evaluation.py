"""
CivicLens Evaluation Module (Phase 5)
Comprehensive classification metrics, robustness drop calculation, and benchmark orchestration.
"""
import os
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)

def evaluate_classification_predictions(
    y_true: Union[List[str], np.ndarray],
    y_pred: Union[List[str], np.ndarray],
    class_labels: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Computes standard NLP classification metrics including Macro-F1, Weighted-F1,
    per-class precision/recall, and confusion matrix.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    if class_labels is None:
        class_labels = sorted(list(set(y_true).union(set(y_pred))))
        
    acc = accuracy_score(y_true, y_pred)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=class_labels, average="macro", zero_division=0
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=class_labels, average="weighted", zero_division=0
    )
    
    # Per-class breakdown
    p_per, r_per, f1_per, sup_per = precision_recall_fscore_support(
        y_true, y_pred, labels=class_labels, average=None, zero_division=0
    )
    
    cm = confusion_matrix(y_true, y_pred, labels=class_labels)
    
    per_class_metrics = {}
    for i, label in enumerate(class_labels):
        per_class_metrics[label] = {
            "precision": float(p_per[i]),
            "recall": float(r_per[i]),
            "f1_score": float(f1_per[i]),
            "support": int(sup_per[i])
        }
        
    return {
        "accuracy": float(acc),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "labels": class_labels,
        "confusion_matrix": cm.tolist(),
        "per_class": per_class_metrics
    }

def calculate_code_mixed_robustness(
    standard_true: List[str],
    standard_pred: List[str],
    codemix_pred: List[str]
) -> Dict[str, Any]:
    """
    Measures the performance difference (robustness drop) when complaints are presented
    in code-mixed Indian dialect vs standard English.
    
    Delta F1 = F1(standard) - F1(codemix)
    Delta Accuracy = Acc(standard) - Acc(codemix)
    """
    std_metrics = evaluate_classification_predictions(standard_true, standard_pred)
    cm_metrics = evaluate_classification_predictions(standard_true, codemix_pred)
    
    delta_f1 = std_metrics["macro_f1"] - cm_metrics["macro_f1"]
    delta_acc = std_metrics["accuracy"] - cm_metrics["accuracy"]
    
    # Identify flipped predictions (cases correctly predicted in English but failed in code-mix)
    flips = []
    for i, (yt, yp_std, yp_cm) in enumerate(zip(standard_true, standard_pred, codemix_pred)):
        if yp_std == yt and yp_cm != yt:
            flips.append({
                "index": i,
                "ground_truth": yt,
                "predicted_standard": yp_std,
                "predicted_codemix": yp_cm
            })
            
    return {
        "standard_accuracy": std_metrics["accuracy"],
        "codemix_accuracy": cm_metrics["accuracy"],
        "delta_accuracy": float(delta_acc),
        "standard_macro_f1": std_metrics["macro_f1"],
        "codemix_macro_f1": cm_metrics["macro_f1"],
        "delta_macro_f1": float(delta_f1),
        "vulnerability_flip_rate": len(flips) / len(standard_true) if standard_true else 0.0,
        "flipped_cases": flips
    }
