"""
CivicLens Benchmark Runner (Phase 5)
Executes comprehensive experimental evaluations, generates comparison tables,
produces publication-grade figures, and serializes trained models.
"""
import os
import sys

# Ensure repository root is in python path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import pandas as pd
import numpy as np
from src.preprocessing import clean_text, validate_complaints_data, grouped_train_test_split
from src.classification import (
    TfidfBaselineClassifier,
    SemanticEmbeddingClassifier,
    save_model
)
from src.similarity import HybridSimilarityEngine
from src.evaluation import (
    evaluate_classification_predictions,
    calculate_code_mixed_robustness
)
from src.visualization import (
    plot_confusion_matrix,
    plot_robustness_comparison,
    plot_similarity_score_distributions
)

def run_all_benchmarks(
    complaints_path: str = "data/sample_complaints.csv",
    pairs_path: str = "data/complaint_pairs.csv",
    code_mixed_path: str = "data/code_mixed_eval.csv",
    results_dir: str = "results",
    models_dir: str = "models"
) -> pd.DataFrame:
    print("=" * 60)
    print("CivicLens: Starting Comprehensive NLP Benchmark Suite")
    print("=" * 60)
    
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(os.path.join(results_dir, "figures"), exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    # 1. Load and validate data
    print("\n[1/5] Loading and Preprocessing Datasets...")
    df_raw = pd.read_csv(complaints_path)
    df_complaints = validate_complaints_data(df_raw)
    df_pairs = pd.read_csv(pairs_path)
    df_cm = pd.read_csv(code_mixed_path)
    
    # Leak-free split
    train_df, test_df = grouped_train_test_split(
        df_complaints, group_col="incident_group_id", test_size=0.25, random_state=42
    )
    print(f"Training instances: {len(train_df)} | Test instances: {len(test_df)}")
    
    # 2. Train and Evaluate Classification Models
    print("\n[2/5] Training Classification Models...")
    y_test = test_df["category"].values
    
    # Baseline 1: TF-IDF + Logistic Regression
    print(" - Training TF-IDF (Word n-grams) + Logistic Regression...")
    tfidf_lr = TfidfBaselineClassifier(classifier_type="logistic_regression", ngram_range=(1, 2))
    tfidf_lr.fit(train_df["text"].tolist(), train_df["category"].tolist())
    pred_tfidf_lr = tfidf_lr.predict(test_df["text"].tolist())
    metrics_tfidf_lr = evaluate_classification_predictions(y_test, pred_tfidf_lr)
    
    # Baseline 2: TF-IDF (Char n-grams) + Linear SVM (Calibrated)
    print(" - Training TF-IDF (Char n-grams) + Calibrated Linear SVM...")
    tfidf_svm = TfidfBaselineClassifier(
        classifier_type="linear_svm", analyzer="char_wb", ngram_range=(3, 5)
    )
    tfidf_svm.fit(train_df["text"].tolist(), train_df["category"].tolist())
    pred_tfidf_svm = tfidf_svm.predict(test_df["text"].tolist())
    metrics_tfidf_svm = evaluate_classification_predictions(y_test, pred_tfidf_svm)
    
    # Semantic: Multilingual Sentence Transformer + Logistic Regression
    print(" - Training Multilingual Sentence Transformer (paraphrase-multilingual-MiniLM-L12-v2)...")
    semantic_clf = SemanticEmbeddingClassifier()
    semantic_clf.fit(train_df["text"].tolist(), train_df["category"].tolist())
    pred_semantic = semantic_clf.predict(test_df["text"].tolist())
    metrics_semantic = evaluate_classification_predictions(y_test, pred_semantic)
    
    # Fit production serving models on full validated complaints corpus
    print(" - Fitting production models on full complaints corpus for serving...")
    prod_tfidf_lr = TfidfBaselineClassifier(classifier_type="logistic_regression", ngram_range=(1, 2))
    prod_tfidf_lr.fit(df_complaints["text"].tolist(), df_complaints["category"].tolist())
    
    prod_tfidf_svm = TfidfBaselineClassifier(
        classifier_type="linear_svm", analyzer="char_wb", ngram_range=(3, 5)
    )
    prod_tfidf_svm.fit(df_complaints["text"].tolist(), df_complaints["category"].tolist())
    
    prod_semantic = SemanticEmbeddingClassifier(C=5.0, temperature=0.35)
    prod_semantic.fit(df_complaints["text"].tolist(), df_complaints["category"].tolist())

    # Save trained production models
    save_model(prod_tfidf_lr, os.path.join(models_dir, "tfidf_baseline_lr.joblib"))
    save_model(prod_tfidf_svm, os.path.join(models_dir, "tfidf_char_svm.joblib"))
    save_model(prod_semantic, os.path.join(models_dir, "semantic_classifier.joblib"))
    print(f"Model artifacts saved to {models_dir}/")
    
    # 3. Evaluate Code-Mixed Linguistic Robustness
    print("\n[3/5] Evaluating Linguistic Robustness (English vs Code-Mixed Parallel Pairs)...")
    cm_true = df_cm["category"].tolist()
    cm_english_texts = df_cm["standard_english"].tolist()
    cm_variant_texts = df_cm["code_mixed_variant"].tolist()
    
    # Test all 3 models on English vs Code-mixed
    rob_lr = calculate_code_mixed_robustness(
        cm_true,
        tfidf_lr.predict(cm_english_texts),
        tfidf_lr.predict(cm_variant_texts)
    )
    rob_svm = calculate_code_mixed_robustness(
        cm_true,
        tfidf_svm.predict(cm_english_texts),
        tfidf_svm.predict(cm_variant_texts)
    )
    rob_sem = calculate_code_mixed_robustness(
        cm_true,
        semantic_clf.predict(cm_english_texts),
        semantic_clf.predict(cm_variant_texts)
    )
    
    # 4. Evaluate Duplicate Detection (Lexical vs Semantic vs Hybrid)
    print("\n[4/5] Evaluating Duplicate Detection on Labeled Pairs...")
    sim_engine = HybridSimilarityEngine()
    sim_engine.fit_corpus(df_complaints["text"].tolist())
    
    # Lexical only (alpha = 0.0)
    eval_lex = sim_engine.evaluate_pairs(df_pairs, alpha=0.0, threshold=0.25)
    # Semantic only (alpha = 1.0)
    eval_sem = sim_engine.evaluate_pairs(df_pairs, alpha=1.0, threshold=0.60)
    # Hybrid (alpha = 0.7)
    opt_tau, _ = sim_engine.optimize_threshold(df_pairs, alpha=0.70)
    eval_hyb = sim_engine.evaluate_pairs(df_pairs, alpha=0.70, threshold=opt_tau)
    
    # 5. Compile Metrics Table and Figures
    print("\n[5/5] Compiling Results and Generating Publication Figures...")
    
    metrics_records = [
        {
            "Task": "Classification",
            "Model": "TF-IDF (Word) + Logistic Regression",
            "Accuracy": round(metrics_tfidf_lr["accuracy"], 4),
            "Macro_F1": round(metrics_tfidf_lr["macro_f1"], 4),
            "Weighted_F1": round(metrics_tfidf_lr["weighted_f1"], 4),
            "CodeMixed_Robustness_Drop_F1": round(rob_lr["delta_macro_f1"], 4)
        },
        {
            "Task": "Classification",
            "Model": "TF-IDF (Char WB) + Calibrated Linear SVM",
            "Accuracy": round(metrics_tfidf_svm["accuracy"], 4),
            "Macro_F1": round(metrics_tfidf_svm["macro_f1"], 4),
            "Weighted_F1": round(metrics_tfidf_svm["weighted_f1"], 4),
            "CodeMixed_Robustness_Drop_F1": round(rob_svm["delta_macro_f1"], 4)
        },
        {
            "Task": "Classification",
            "Model": "Multilingual MiniLM + Softmax Head",
            "Accuracy": round(metrics_semantic["accuracy"], 4),
            "Macro_F1": round(metrics_semantic["macro_f1"], 4),
            "Weighted_F1": round(metrics_semantic["weighted_f1"], 4),
            "CodeMixed_Robustness_Drop_F1": round(rob_sem["delta_macro_f1"], 4)
        },
        {
            "Task": "Deduplication",
            "Model": "Lexical Jaccard/Char TF-IDF (alpha = 0.0)",
            "Accuracy": round(eval_lex["accuracy"], 4),
            "Precision": round(eval_lex["precision"], 4),
            "Recall": round(eval_lex["recall"], 4),
            "Macro_F1": round(eval_lex["f1_score"], 4),
            "False_Positive_Rate": round(eval_lex["false_positive_rate"], 4)
        },
        {
            "Task": "Deduplication",
            "Model": "Multilingual Dense Cosine (alpha = 1.0)",
            "Accuracy": round(eval_sem["accuracy"], 4),
            "Precision": round(eval_sem["precision"], 4),
            "Recall": round(eval_sem["recall"], 4),
            "Macro_F1": round(eval_sem["f1_score"], 4),
            "False_Positive_Rate": round(eval_sem["false_positive_rate"], 4)
        },
        {
            "Task": "Deduplication",
            "Model": f"Hybrid Lexical-Semantic (alpha = 0.70, tau* = {opt_tau:.2f})",
            "Accuracy": round(eval_hyb["accuracy"], 4),
            "Precision": round(eval_hyb["precision"], 4),
            "Recall": round(eval_hyb["recall"], 4),
            "Macro_F1": round(eval_hyb["f1_score"], 4),
            "False_Positive_Rate": round(eval_hyb["false_positive_rate"], 4)
        }
    ]
    
    metrics_df = pd.DataFrame(metrics_records)
    metrics_csv_path = os.path.join(results_dir, "metrics.csv")
    metrics_df.to_csv(metrics_csv_path, index=False)
    print(f"Metrics table exported to {metrics_csv_path}")
    
    # Save Figures
    plot_confusion_matrix(
        metrics_semantic["confusion_matrix"],
        labels=metrics_semantic["labels"],
        title="Multilingual Semantic Classifier - Confusion Matrix",
        save_path=os.path.join(results_dir, "figures", "confusion_matrix_semantic.png")
    )
    
    plot_confusion_matrix(
        metrics_tfidf_lr["confusion_matrix"],
        labels=metrics_tfidf_lr["labels"],
        title="TF-IDF Baseline - Confusion Matrix",
        save_path=os.path.join(results_dir, "figures", "confusion_matrix_baseline.png")
    )
    
    rob_data = [
        {
            "Model": "TF-IDF + LR",
            "English_F1": rob_lr["standard_macro_f1"],
            "CodeMixed_F1": rob_lr["codemix_macro_f1"]
        },
        {
            "Model": "TF-IDF Char + SVM",
            "English_F1": rob_svm["standard_macro_f1"],
            "CodeMixed_F1": rob_svm["codemix_macro_f1"]
        },
        {
            "Model": "Multilingual MiniLM",
            "English_F1": rob_sem["standard_macro_f1"],
            "CodeMixed_F1": rob_sem["codemix_macro_f1"]
        }
    ]
    plot_robustness_comparison(
        rob_data,
        save_path=os.path.join(results_dir, "figures", "robustness_comparison.png")
    )
    
    # Compute similarity scores for duplicate vs non-duplicate distribution plot
    dup_scores = []
    non_dup_scores = []
    for _, row in df_pairs.iterrows():
        s = sim_engine.compute_hybrid_similarity(row["complaint_1"], row["complaint_2"], alpha=0.70)["hybrid_similarity"]
        if int(row["is_duplicate"]) == 1:
            dup_scores.append(s)
        else:
            non_dup_scores.append(s)
            
    plot_similarity_score_distributions(
        dup_scores,
        non_dup_scores,
        optimal_threshold=opt_tau,
        save_path=os.path.join(results_dir, "figures", "similarity_distributions.png")
    )
    
    print("\nBenchmark completed successfully! All artifacts generated.")
    return metrics_df

if __name__ == "__main__":
    df_res = run_all_benchmarks()
    print("\n=== Benchmark Results Summary ===")
    print(df_res.to_string())
