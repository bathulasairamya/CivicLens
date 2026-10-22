"""
Unit tests for CivicLens evaluation module (Phase 5).
"""
import pytest
import numpy as np
from src.evaluation import (
    evaluate_classification_predictions,
    calculate_code_mixed_robustness
)

def test_evaluate_classification_predictions_perfect():
    y_true = ["Water supply", "Electricity", "Sanitation", "Roads"]
    y_pred = ["Water supply", "Electricity", "Sanitation", "Roads"]
    metrics = evaluate_classification_predictions(y_true, y_pred)
    
    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert metrics["weighted_f1"] == 1.0
    assert len(metrics["confusion_matrix"]) == 4

def test_evaluate_classification_predictions_imperfect():
    y_true = ["Water supply", "Water supply", "Electricity", "Electricity"]
    y_pred = ["Water supply", "Electricity", "Electricity", "Electricity"]
    metrics = evaluate_classification_predictions(y_true, y_pred)
    
    assert metrics["accuracy"] == 0.75
    assert metrics["macro_f1"] < 1.0
    assert "Water supply" in metrics["per_class"]
    assert "Electricity" in metrics["per_class"]

def test_calculate_code_mixed_robustness():
    # Scenario: model predicts correctly on English, but makes 1 error on code-mix
    standard_true = ["Water supply", "Electricity", "Roads"]
    standard_pred = ["Water supply", "Electricity", "Roads"]
    codemix_pred = ["Water supply", "Sanitation", "Roads"]  # Electricity flipped to Sanitation
    
    rob = calculate_code_mixed_robustness(standard_true, standard_pred, codemix_pred)
    assert rob["standard_accuracy"] == 1.0
    assert rob["codemix_accuracy"] < 1.0
    assert rob["delta_accuracy"] > 0.0
    assert rob["delta_macro_f1"] > 0.0
    assert len(rob["flipped_cases"]) == 1
    assert rob["flipped_cases"][0]["ground_truth"] == "Electricity"
    assert rob["flipped_cases"][0]["predicted_codemix"] == "Sanitation"
