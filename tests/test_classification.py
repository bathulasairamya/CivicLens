"""
Unit tests for CivicLens classification module (Phase 3).
"""
import os
import pytest
import numpy as np
from src.classification import (
    TfidfBaselineClassifier,
    SemanticEmbeddingClassifier,
    calculate_normalized_entropy,
    save_model,
    load_model
)

@pytest.fixture
def sample_training_data():
    texts = [
        "Major pipeline leak in sector 2 drinking water cut off",
        "No tap water pressure in morning block C taps dry",
        "Power failure transformer blast complete blackout dark",
        "High voltage fluctuation fridge damaged electrical fire wire",
        "Garbage pile on street stinks open drainage trash",
        "Kachra nahi uthaya gaya sanitation issue overflowing bin"
    ]
    labels = [
        "Water supply",
        "Water supply",
        "Electricity",
        "Electricity",
        "Sanitation",
        "Sanitation"
    ]
    return texts, labels

def test_normalized_entropy_extremes():
    # Certain distribution: entropy should be 0.0
    certain_p = np.array([1.0, 0.0, 0.0])
    assert calculate_normalized_entropy(certain_p) < 1e-4
    
    # Uniform distribution: entropy should be 1.0
    uniform_p = np.array([1/3, 1/3, 1/3])
    assert abs(calculate_normalized_entropy(uniform_p) - 1.0) < 1e-4

def test_tfidf_baseline_logistic_regression(sample_training_data):
    texts, labels = sample_training_data
    clf = TfidfBaselineClassifier(classifier_type="logistic_regression", ngram_range=(1, 2))
    clf.fit(texts, labels)
    
    preds = clf.predict(["pipeline leak no water", "transformer sparking blackout"])
    assert len(preds) == 2
    assert preds[0] in clf.classes_
    assert preds[1] in clf.classes_
    
    probas = clf.predict_proba(["pipeline leak"])
    assert probas.shape == (1, len(clf.classes_))
    assert np.isclose(np.sum(probas), 1.0)

def test_tfidf_baseline_linear_svm(sample_training_data):
    texts, labels = sample_training_data
    clf = TfidfBaselineClassifier(
        classifier_type="linear_svm", analyzer="char_wb", ngram_range=(3, 5)
    )
    clf.fit(texts, labels)
    
    preds = clf.predict(["water pipe issue", "power outage"])
    assert len(preds) == 2
    assert preds[0] in clf.classes_
    
    # Calibrated probabilities check
    probas = clf.predict_proba(["water pipe"])
    assert probas.shape == (1, len(clf.classes_))
    assert np.isclose(np.sum(probas), 1.0)

def test_predict_with_uncertainty(sample_training_data):
    texts, labels = sample_training_data
    clf = TfidfBaselineClassifier(classifier_type="logistic_regression")
    clf.fit(texts, labels)
    
    results = clf.predict_with_uncertainty(
        ["unrelated nonsense token xyz 123", "pipeline burst drinking water contamination"],
        conf_threshold=0.65,
        entropy_threshold=0.60
    )
    assert len(results) == 2
    # The vague input should trigger human review
    assert results[0]["is_uncertain"] is True
    assert results[0]["review_flag"] == "FLAGGED_FOR_HUMAN_REVIEW"
    assert len(results[0]["reasons"]) > 0

def test_semantic_classifier_fit_and_predict(sample_training_data):
    texts, labels = sample_training_data
    clf = SemanticEmbeddingClassifier()
    clf.fit(texts, labels)
    
    preds = clf.predict(["Tap water pipeline leak in sector", "Power failure blackout in area"])
    assert len(preds) == 2
    assert preds[0] == "Water supply"
    assert preds[1] == "Electricity"

def test_model_save_and_load(tmp_path, sample_training_data):
    texts, labels = sample_training_data
    clf = TfidfBaselineClassifier(classifier_type="logistic_regression")
    clf.fit(texts, labels)
    
    filepath = os.path.join(tmp_path, "test_model.joblib")
    save_model(clf, filepath)
    loaded_clf = load_model(filepath)
    
    assert loaded_clf is not None
    orig_preds = clf.predict(texts[:2])
    loaded_preds = loaded_clf.predict(texts[:2])
    np.testing.assert_array_equal(orig_preds, loaded_preds)
