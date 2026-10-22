"""
Unit tests for CivicLens preprocessing pipeline (Phase 2).
"""
import pytest
import pandas as pd
from src.preprocessing import (
    clean_text,
    detect_script_characteristics,
    validate_complaints_data,
    grouped_train_test_split,
    VALID_CATEGORIES
)

def test_clean_text_normalizes_whitespace_and_punctuation():
    raw = "   Water pipe   burst near Gate 3!!!!!   "
    cleaned = clean_text(raw)
    assert cleaned == "water pipe burst near gate 3!"

def test_clean_text_removes_urls():
    raw = "Check this incident https://example.com/item at main road"
    cleaned = clean_text(raw)
    assert "https" not in cleaned
    assert "at main road" in cleaned

def test_detect_script_characteristics():
    hindi_text = "पानी की समस्या है"
    stats_hi = detect_script_characteristics(hindi_text)
    assert stats_hi["has_devanagari"] is True
    assert stats_hi["primary_script"] == "Devanagari"
    
    eng_text = "Heavy traffic jam on 4th block"
    stats_en = detect_script_characteristics(eng_text)
    assert stats_en["has_latin"] is True
    assert "Latin" in stats_en["primary_script"]

def test_validate_complaints_data_valid():
    df = pd.DataFrame({
        "complaint_id": ["C1", "C2"],
        "text": ["Broken streetlight", "Leaking pipe"],
        "category": ["Roads", "Water supply"]
    })
    validated = validate_complaints_data(df)
    assert len(validated) == 2
    assert "text" in validated.columns

def test_validate_complaints_data_invalid_category():
    df = pd.DataFrame({
        "complaint_id": ["C1"],
        "text": ["Alien invasion"],
        "category": ["Astronomy"]
    })
    with pytest.raises(ValueError, match="Found invalid categories"):
        validate_complaints_data(df)

def test_grouped_train_test_split_no_leakage():
    df = pd.DataFrame({
        "complaint_id": [f"C{i}" for i in range(10)],
        "text": [f"Complaint {i}" for i in range(10)],
        "category": ["Roads"] * 5 + ["Water supply"] * 5,
        "incident_group_id": ["G1", "G1", "G2", "G2", "G3", "G3", "G4", "G4", "G5", "G5"]
    })
    train_df, test_df = grouped_train_test_split(df, group_col="incident_group_id", test_size=0.4, random_state=42)
    
    train_groups = set(train_df["incident_group_id"])
    test_groups = set(test_df["incident_group_id"])
    
    # Critical test: zero intersection between train and test groups
    assert len(train_groups.intersection(test_groups)) == 0
    assert len(train_df) + len(test_df) == 10

def test_dataset_files_integrity():
    """Validates that all three created dataset files load and comply with schema."""
    df_complaints = pd.read_csv("data/sample_complaints.csv")
    assert len(df_complaints) >= 50
    validated = validate_complaints_data(df_complaints)
    assert len(validated) == len(df_complaints)
    
    df_pairs = pd.read_csv("data/complaint_pairs.csv")
    assert {"pair_id", "complaint_1", "complaint_2", "is_duplicate", "pair_type"}.issubset(df_pairs.columns)
    assert set(df_pairs["is_duplicate"].unique()).issubset({0, 1})
    
    df_eval = pd.read_csv("data/code_mixed_eval.csv")
    assert {"eval_id", "category", "language", "standard_english", "code_mixed_variant"}.issubset(df_eval.columns)
