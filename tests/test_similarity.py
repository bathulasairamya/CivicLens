"""
Unit tests for CivicLens similarity and deduplication engine (Phase 4).
"""
import pytest
import pandas as pd
from src.similarity import (
    token_jaccard_similarity,
    char_ngram_jaccard_similarity,
    HybridSimilarityEngine
)

def test_jaccard_identical_text():
    text = "water pipeline burst on gate 3"
    assert token_jaccard_similarity(text, text) == 1.0
    assert char_ngram_jaccard_similarity(text, text) == 1.0

def test_jaccard_disjoint_text():
    t1 = "water pipeline leak"
    t2 = "electricity blackout transformer"
    assert token_jaccard_similarity(t1, t2) == 0.0

def test_jaccard_empty_text():
    assert token_jaccard_similarity("", "some text") == 0.0

def test_hybrid_similarity_identical():
    engine = HybridSimilarityEngine()
    text = "water pipe broken in front of hostel"
    res = engine.compute_hybrid_similarity(text, text, alpha=0.7)
    assert res["lexical_similarity"] >= 0.99
    assert res["semantic_similarity"] >= 0.99
    assert res["hybrid_similarity"] >= 0.99

def test_hybrid_similarity_cross_category():
    engine = HybridSimilarityEngine()
    t1 = "water pipeline burst road flooded"
    t2 = "bus ticket machine broken conductor misbehaving"
    res = engine.compute_hybrid_similarity(t1, t2, alpha=0.7)
    assert res["hybrid_similarity"] < 0.45

def test_predict_pair_duplicate():
    engine = HybridSimilarityEngine()
    t1 = "Massive pothole on highway causing vehicle damage"
    t2 = "Huge crater on the highway cars are getting damaged"
    pred = engine.predict_pair(t1, t2, alpha=0.7, threshold=0.55)
    assert "is_duplicate" in pred
    assert "hybrid_similarity" in pred
    assert pred["is_duplicate"] is True
    assert pred["verdict"] == "DUPLICATE_REPORT"

def test_find_top_duplicates():
    engine = HybridSimilarityEngine()
    corpus = [
        "Water leak near entrance gate",
        "Transformer blast on avenue 4",
        "Pipe burst flooding road near entrance"
    ]
    query = "Water pipeline ruptured near entrance"
    top_matches = engine.find_top_duplicates(query, corpus, top_k=2, alpha=0.7)
    assert len(top_matches) == 2
    # The water related candidates should rank above the transformer blast
    assert "Transformer" not in top_matches[0]["text"]
    assert top_matches[0]["hybrid_similarity"] >= top_matches[1]["hybrid_similarity"]

def test_evaluate_pairs_and_optimize():
    engine = HybridSimilarityEngine()
    pairs_data = pd.DataFrame({
        "complaint_1": [
            "water pipe broken",
            "blackout in ward 12",
            "pothole on mg road"
        ],
        "complaint_2": [
            "broken pipeline no water",
            "transformer exploded power outage",
            "completely unrelated student hostel issue"
        ],
        "is_duplicate": [1, 1, 0]
    })
    res = engine.evaluate_pairs(pairs_data, alpha=0.7, threshold=0.50)
    assert "f1_score" in res
    assert "accuracy" in res
    assert res["f1_score"] > 0.0
    
    opt_tau, best_f1 = engine.optimize_threshold(pairs_data, alpha=0.7)
    assert 0.30 <= opt_tau <= 0.90
    assert best_f1 > 0.0
