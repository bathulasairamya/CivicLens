"""
CivicLens Preprocessing Pipeline
Text cleaning, script detection, data validation, and leakage-proof dataset splitting.
"""
import re
import unicodedata
from typing import List, Tuple, Dict, Any, Optional
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit

VALID_CATEGORIES: List[str] = [
    "Water supply",
    "Electricity",
    "Sanitation",
    "Roads",
    "Transport",
    "Campus facilities"
]

def clean_text(text: Optional[str], lowercase: bool = True) -> str:
    """
    Cleans raw complaint text while preserving code-mixing cues, multilingual characters,
    and negation indicators.
    
    Args:
        text: Input string.
        lowercase: Whether to convert text to lower case.
        
    Returns:
        Cleaned, normalized string.
    """
    if text is None or not isinstance(text, str):
        return ""
    
    # Normalize unicode forms (NFKC handles ligatures and compatibility characters)
    text = unicodedata.normalize("NFKC", text)
    
    # Replace URLs and email addresses with standard tokens if present
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"\S+@\S+", "", text)
    
    # Normalize excessive punctuation (e.g., "????" -> "?", "!!!!" -> "!")
    text = re.sub(r"([!?.,])\1+", r"\1", text)
    
    # Collapse multiple whitespaces, tabs, and newlines into single spaces
    text = re.sub(r"\s+", " ", text).strip()
    
    if lowercase:
        text = text.lower()
        
    return text

def detect_script_characteristics(text: str) -> Dict[str, Any]:
    """
    Analyzes character ranges to determine the script and presence of non-Latin characters.
    
    Args:
        text: Input complaint text.
        
    Returns:
        Dictionary with script analysis flags and statistics.
    """
    if not text:
        return {"has_indic_script": False, "primary_script": "empty", "char_count": 0}
        
    has_devanagari = any('\u0900' <= char <= '\u097F' for char in text)
    has_tamil = any('\u0B80' <= char <= '\u0BFF' for char in text)
    has_telugu = any('\u0C00' <= char <= '\u0C7F' for char in text)
    has_latin = any('a' <= char.lower() <= 'z' for char in text)
    
    if has_devanagari:
        primary = "Devanagari"
    elif has_tamil:
        primary = "Tamil"
    elif has_telugu:
        primary = "Telugu"
    elif has_latin:
        primary = "Latin (English/Romanized Code-Mixed)"
    else:
        primary = "Other"
        
    return {
        "has_indic_script": has_devanagari or has_tamil or has_telugu,
        "has_devanagari": has_devanagari,
        "has_tamil": has_tamil,
        "has_telugu": has_telugu,
        "has_latin": has_latin,
        "primary_script": primary,
        "char_count": len(text)
    }

def validate_complaints_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validates complaint DataFrame schema, categories, and null values.
    
    Args:
        df: Pandas DataFrame containing complaints.
        
    Returns:
        Validated and cleaned DataFrame.
    """
    required_cols = {"complaint_id", "text", "category"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in dataset: {missing}")
        
    # Drop rows with empty text
    df_clean = df.copy()
    df_clean["text"] = df_clean["text"].astype(str).apply(clean_text)
    df_clean = df_clean[df_clean["text"].str.len() > 0]
    
    # Check invalid categories
    invalid_cats = set(df_clean["category"].unique()) - set(VALID_CATEGORIES)
    if invalid_cats:
        raise ValueError(f"Found invalid categories in dataset: {invalid_cats}")
        
    return df_clean

def grouped_train_test_split(
    df: pd.DataFrame,
    group_col: str = "incident_group_id",
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits complaints into train and test sets such that all variants of an incident
    remain entirely within one split, preventing duplicate data leakage.
    
    Args:
        df: DataFrame containing complaints and grouping column.
        group_col: Column name identifying the incident cluster.
        test_size: Proportion of groups to include in the test split.
        random_state: Seed for reproducibility.
        
    Returns:
        Tuple of (train_df, test_df).
    """
    if group_col not in df.columns:
        # Fallback to stratified random split if no incident group ID exists
        from sklearn.model_selection import train_test_split
        return train_test_split(df, test_size=test_size, random_state=random_state, stratify=df["category"])
        
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(gss.split(df, groups=df[group_col]))
    
    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)
    return train_df, test_df
