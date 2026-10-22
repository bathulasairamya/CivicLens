"""
CivicLens Deduplication & Similarity Module (Phase 4)
Implements lexical, semantic, and hybrid similarity algorithms for duplicate incident detection,
top-K candidate retrieval, and validation threshold calibration.
"""
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sentence_transformers import SentenceTransformer

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

def token_jaccard_similarity(text1: str, text2: str) -> float:
    """Computes Jaccard similarity between word tokens of two texts."""
    tokens1 = set(text1.lower().split())
    tokens2 = set(text2.lower().split())
    if not tokens1 or not tokens2:
        return 0.0
    intersection = tokens1.intersection(tokens2)
    union = tokens1.union(tokens2)
    return len(intersection) / len(union)

def char_ngram_jaccard_similarity(text1: str, text2: str, n: int = 3) -> float:
    """Computes character n-gram Jaccard similarity, effective for noisy/code-mixed transliteration."""
    t1 = text1.lower()
    t2 = text2.lower()
    if len(t1) < n or len(t2) < n:
        return 1.0 if t1 == t2 else 0.0
    ngrams1 = set(t1[i:i+n] for i in range(len(t1) - n + 1))
    ngrams2 = set(t2[i:i+n] for i in range(len(t2) - n + 1))
    return len(ngrams1.intersection(ngrams2)) / len(ngrams1.union(ngrams2))

class HybridSimilarityEngine:
    """
    Combines lexical character/subword patterns with deep multilingual sentence embeddings
    to detect duplicate and paraphrased complaints across languages.
    """
    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        device: Optional[str] = None
    ):
        self.model_name = model_name
        self.device = device
        self._embedder: Optional[SentenceTransformer] = None
        self.vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)
        self._is_vectorizer_fitted = False
        
    @property
    def embedder(self) -> SentenceTransformer:
        if self._embedder is None:
            self._embedder = SentenceTransformer(self.model_name, device=self.device)
        return self._embedder

    def fit_corpus(self, corpus_texts: List[str]) -> "HybridSimilarityEngine":
        """Fits TF-IDF vectorizer on background corpus if available."""
        if corpus_texts:
            self.vectorizer.fit(corpus_texts)
            self._is_vectorizer_fitted = True
        return self

    def compute_lexical_similarity(self, text1: str, text2: str) -> float:
        """Computes lexical similarity using TF-IDF or character n-gram fallback."""
        if self._is_vectorizer_fitted:
            vecs = self.vectorizer.transform([text1, text2])
            sim = float(cosine_similarity(vecs[0], vecs[1])[0][0])
            return max(0.0, min(1.0, sim))
        else:
            # Fallback to blended word and char-ngram Jaccard
            w_sim = token_jaccard_similarity(text1, text2)
            c_sim = char_ngram_jaccard_similarity(text1, text2, n=3)
            return 0.4 * w_sim + 0.6 * c_sim

    def compute_semantic_similarity(self, text1: str, text2: str) -> float:
        """Computes multilingual cosine similarity using dense sentence embeddings."""
        embs = self.embedder.encode([text1, text2], convert_to_numpy=True, show_progress_bar=False)
        sim = float(cosine_similarity([embs[0]], [embs[1]])[0][0])
        return max(0.0, min(1.0, sim))

    def compute_hybrid_similarity(
        self,
        text1: str,
        text2: str,
        alpha: float = 0.70
    ) -> Dict[str, float]:
        """
        Calculates hybrid similarity: Sim = alpha * Sim_sem + (1 - alpha) * Sim_lex
        """
        alpha = max(0.0, min(1.0, alpha))
        lex_sim = self.compute_lexical_similarity(text1, text2)
        sem_sim = self.compute_semantic_similarity(text1, text2)
        hybrid_sim = (alpha * sem_sim) + ((1.0 - alpha) * lex_sim)
        
        return {
            "lexical_similarity": float(lex_sim),
            "semantic_similarity": float(sem_sim),
            "hybrid_similarity": float(hybrid_sim),
            "alpha": float(alpha)
        }

    def predict_pair(
        self,
        text1: str,
        text2: str,
        alpha: float = 0.70,
        threshold: float = 0.60
    ) -> Dict[str, Any]:
        """Classifies whether two complaint texts are duplicates under a given threshold."""
        scores = self.compute_hybrid_similarity(text1, text2, alpha=alpha)
        is_dup = bool(scores["hybrid_similarity"] >= threshold)
        
        return {
            **scores,
            "threshold": threshold,
            "is_duplicate": is_dup,
            "verdict": "DUPLICATE_REPORT" if is_dup else "UNIQUE_INCIDENT"
        }

    def find_top_duplicates(
        self,
        query_text: str,
        candidates: List[str],
        candidate_ids: Optional[List[str]] = None,
        candidate_categories: Optional[List[str]] = None,
        top_k: int = 5,
        alpha: float = 0.70,
        threshold: float = 0.50
    ) -> List[Dict[str, Any]]:
        """
        Retrieves and ranks the top-K most similar candidate complaints.
        """
        if not candidates:
            return []
            
        q_emb = self.embedder.encode([query_text], convert_to_numpy=True, show_progress_bar=False)[0]
        cand_embs = self.embedder.encode(candidates, convert_to_numpy=True, show_progress_bar=False)
        sem_scores = cosine_similarity([q_emb], cand_embs)[0]
        
        ranked_results = []
        for i, text in enumerate(candidates):
            sem_s = float(sem_scores[i])
            lex_s = self.compute_lexical_similarity(query_text, text)
            hybrid_s = (alpha * sem_s) + ((1.0 - alpha) * lex_s)
            
            c_id = candidate_ids[i] if candidate_ids and i < len(candidate_ids) else f"CAND-{i+1:03d}"
            c_cat = candidate_categories[i] if candidate_categories and i < len(candidate_categories) else "Unspecified"
            
            ranked_results.append({
                "candidate_id": c_id,
                "text": text,
                "category": c_cat,
                "hybrid_similarity": float(hybrid_s),
                "semantic_similarity": float(sem_s),
                "lexical_similarity": float(lex_s),
                "is_potential_duplicate": bool(hybrid_s >= threshold)
            })
            
        ranked_results.sort(key=lambda x: x["hybrid_similarity"], reverse=True)
        return ranked_results[:top_k]

    def evaluate_pairs(
        self,
        pairs_df: pd.DataFrame,
        alpha: float = 0.70,
        threshold: float = 0.60
    ) -> Dict[str, float]:
        """
        Evaluates duplicate detection performance over a labeled pairs DataFrame.
        Columns required: 'complaint_1', 'complaint_2', 'is_duplicate'
        """
        y_true = pairs_df["is_duplicate"].astype(int).values
        y_scores = []
        
        for _, row in pairs_df.iterrows():
            scores = self.compute_hybrid_similarity(row["complaint_1"], row["complaint_2"], alpha=alpha)
            y_scores.append(scores["hybrid_similarity"])
            
        y_scores = np.array(y_scores)
        y_pred = (y_scores >= threshold).astype(int)
        
        p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
        acc = accuracy_score(y_true, y_pred)
        
        # False Positive Rate: FP / (FP + TN)
        fp = int(np.sum((y_pred == 1) & (y_true == 0)))
        tn = int(np.sum((y_pred == 0) & (y_true == 0)))
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        
        return {
            "accuracy": float(acc),
            "precision": float(p),
            "recall": float(r),
            "f1_score": float(f1),
            "false_positive_rate": float(fpr),
            "threshold": float(threshold),
            "alpha": float(alpha)
        }

    def optimize_threshold(
        self,
        pairs_df: pd.DataFrame,
        alpha: float = 0.70,
        thresholds: Optional[np.ndarray] = None
    ) -> Tuple[float, float]:
        """
        Searches for the threshold that maximizes F1 score on validation pairs.
        Precomputes pair similarity scores once for sub-second parameter optimization.
        Returns: (optimal_threshold, best_f1)
        """
        if thresholds is None:
            thresholds = np.linspace(0.30, 0.90, 25)
            
        y_true = pairs_df["is_duplicate"].astype(int).values
        # Precompute hybrid scores once across all pairs
        y_scores = np.array([
            self.compute_hybrid_similarity(row["complaint_1"], row["complaint_2"], alpha=alpha)["hybrid_similarity"]
            for _, row in pairs_df.iterrows()
        ])
        
        best_f1 = -1.0
        best_tau = 0.50
        
        for tau in thresholds:
            y_pred = (y_scores >= tau).astype(int)
            _, _, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
            if f1 > best_f1:
                best_f1 = float(f1)
                best_tau = float(tau)
                
        return best_tau, best_f1
