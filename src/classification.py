"""
CivicLens Classification Module (Phase 3)
Implements lexical baselines (TF-IDF + LR / Calibrated SVM), multilingual sentence embedding
classifiers, and uncertainty quantification for human review triage.
"""
import os
import math
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sentence_transformers import SentenceTransformer

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

def calculate_normalized_entropy(probabilities: np.ndarray) -> float:
    """
    Computes normalized Shannon entropy for a probability distribution.
    Ranges from 0.0 (certain, single class has p=1) to 1.0 (uniform uncertainty across all classes).
    
    H_norm(p) = -sum(p_i * log2(p_i)) / log2(K)
    """
    k = len(probabilities)
    if k <= 1:
        return 0.0
    
    eps = 1e-12
    p = np.clip(probabilities, eps, 1.0)
    # Re-normalize after clip to guarantee valid distribution
    p = p / np.sum(p)
    entropy = -np.sum(p * np.log2(p))
    max_entropy = np.log2(k)
    return float(entropy / max_entropy)

class TfidfBaselineClassifier:
    """
    Lexical baseline complaint classifier using TF-IDF feature extraction
    and Logistic Regression or calibrated Linear Support Vector Machine.
    """
    def __init__(
        self,
        classifier_type: str = "logistic_regression",
        analyzer: str = "word",
        ngram_range: Tuple[int, int] = (1, 2),
        sublinear_tf: bool = True
    ):
        self.classifier_type = classifier_type
        self.analyzer = analyzer
        self.ngram_range = ngram_range
        self.sublinear_tf = sublinear_tf
        self.pipeline: Optional[Pipeline] = None
        self.classes_: Optional[np.ndarray] = None
        
    def fit(self, texts: List[str], labels: List[str]) -> "TfidfBaselineClassifier":
        vectorizer = TfidfVectorizer(
            analyzer=self.analyzer,
            ngram_range=self.ngram_range,
            sublinear_tf=self.sublinear_tf,
            min_df=1
        )
        
        if self.classifier_type == "logistic_regression":
            clf = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
        elif self.classifier_type == "linear_svm":
            # CalibratedClassifierCV wraps LinearSVC to output well-calibrated probabilities
            base_svc = LinearSVC(class_weight="balanced", random_state=42, max_iter=2000)
            from collections import Counter
            min_count = min(Counter(labels).values()) if labels else 1
            cv_splits = max(2, min(3, min_count))
            clf = CalibratedClassifierCV(estimator=base_svc, cv=cv_splits)
        else:
            raise ValueError(f"Unsupported classifier_type: {self.classifier_type}")
            
        self.pipeline = Pipeline([
            ("vectorizer", vectorizer),
            ("classifier", clf)
        ])
        
        self.pipeline.fit(texts, labels)
        self.classes_ = self.pipeline.named_steps["classifier"].classes_
        return self

    def predict(self, texts: List[str]) -> np.ndarray:
        if self.pipeline is None:
            raise RuntimeError("Classifier has not been fitted yet.")
        return self.pipeline.predict(texts)

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        if self.pipeline is None:
            raise RuntimeError("Classifier has not been fitted yet.")
        return self.pipeline.predict_proba(texts)

    def predict_with_uncertainty(
        self,
        texts: List[str],
        conf_threshold: float = 0.55,
        entropy_threshold: float = 0.65
    ) -> List[Dict[str, Any]]:
        probas = self.predict_proba(texts)
        results = []
        
        for p_dist in probas:
            top_class_idx = int(np.argmax(p_dist))
            top_class = str(self.classes_[top_class_idx])
            confidence = float(p_dist[top_class_idx])
            entropy = calculate_normalized_entropy(p_dist)
            
            is_uncertain = (confidence < conf_threshold) or (entropy > entropy_threshold)
            
            reasons = []
            if confidence < conf_threshold:
                reasons.append(f"Confidence {confidence:.2f} below threshold {conf_threshold:.2f}")
            if entropy > entropy_threshold:
                reasons.append(f"Predictive entropy {entropy:.2f} exceeds threshold {entropy_threshold:.2f}")
                
            results.append({
                "predicted_category": top_class,
                "confidence": confidence,
                "normalized_entropy": entropy,
                "is_uncertain": is_uncertain,
                "review_flag": "FLAGGED_FOR_HUMAN_REVIEW" if is_uncertain else "AUTOMATED_ROUTING",
                "reasons": reasons,
                "all_probabilities": {
                    str(c): float(p) for c, p in zip(self.classes_, p_dist)
                }
            })
            
        return results


class SemanticEmbeddingClassifier:
    """
    Semantic multilingual complaint classifier utilizing pretrained Sentence Transformer
    embeddings with a calibrated classification head.
    """
    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        device: Optional[str] = None
    ):
        self.model_name = model_name
        self.device = device
        self._embedder: Optional[SentenceTransformer] = None
        self.clf = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
        self.classes_: Optional[np.ndarray] = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_embedder"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._embedder = None
        
    @property
    def embedder(self) -> SentenceTransformer:
        if self._embedder is None:
            self._embedder = SentenceTransformer(self.model_name, device=self.device)
        return self._embedder

    def extract_embeddings(self, texts: List[str]) -> np.ndarray:
        return self.embedder.encode(texts, convert_to_numpy=True, show_progress_bar=False)

    def fit(self, texts: List[str], labels: List[str]) -> "SemanticEmbeddingClassifier":
        embeddings = self.extract_embeddings(texts)
        self.clf.fit(embeddings, labels)
        self.classes_ = self.clf.classes_
        return self

    def predict(self, texts: List[str]) -> np.ndarray:
        embeddings = self.extract_embeddings(texts)
        return self.clf.predict(embeddings)

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        embeddings = self.extract_embeddings(texts)
        return self.clf.predict_proba(embeddings)

    def predict_with_uncertainty(
        self,
        texts: List[str],
        conf_threshold: float = 0.55,
        entropy_threshold: float = 0.65
    ) -> List[Dict[str, Any]]:
        probas = self.predict_proba(texts)
        results = []
        
        for p_dist in probas:
            top_class_idx = int(np.argmax(p_dist))
            top_class = str(self.classes_[top_class_idx])
            confidence = float(p_dist[top_class_idx])
            entropy = calculate_normalized_entropy(p_dist)
            
            is_uncertain = (confidence < conf_threshold) or (entropy > entropy_threshold)
            
            reasons = []
            if confidence < conf_threshold:
                reasons.append(f"Confidence {confidence:.2f} below threshold {conf_threshold:.2f}")
            if entropy > entropy_threshold:
                reasons.append(f"Predictive entropy {entropy:.2f} exceeds threshold {entropy_threshold:.2f}")
                
            results.append({
                "predicted_category": top_class,
                "confidence": confidence,
                "normalized_entropy": entropy,
                "is_uncertain": is_uncertain,
                "review_flag": "FLAGGED_FOR_HUMAN_REVIEW" if is_uncertain else "AUTOMATED_ROUTING",
                "reasons": reasons,
                "all_probabilities": {
                    str(c): float(p) for c, p in zip(self.classes_, p_dist)
                }
            })
            
        return results


def save_model(model: Any, filepath: str) -> None:
    """Serializes a model object using joblib."""
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    joblib.dump(model, filepath)

def load_model(filepath: str) -> Any:
    """Deserializes a saved model object."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file not found: {filepath}")
    return joblib.load(filepath)
