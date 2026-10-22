"""
CivicLens Manual Interactive Tester
Run with: python test_manual.py
"""
import os
import joblib
import pandas as pd
from src.classification import SemanticEmbeddingClassifier, TfidfBaselineClassifier
from src.similarity import HybridSimilarityEngine

def main():
    print("=" * 60)
    print(" CivicLens: Interactive Manual Tester")
    print("=" * 60)
    
    # 1. Load models
    print("\nLoading trained models and corpus...")
    semantic_model = joblib.load("models/semantic_classifier.joblib")
    tfidf_model = joblib.load("models/tfidf_baseline_lr.joblib")
    
    df_complaints = pd.read_csv("data/sample_complaints.csv")
    similarity_engine = HybridSimilarityEngine()
    similarity_engine.fit_corpus(df_complaints["text"].tolist())
    
    print("Ready!")
    
    while True:
        print("\n" + "-" * 50)
        print("Choose a test mode:")
        print("  1. Classify a Complaint (English / Code-Mixed)")
        print("  2. Check Duplicate Similarity Between Two Complaints")
        print("  3. Find Top-3 Duplicates in Database for a Complaint")
        print("  q. Quit")
        choice = input("Enter choice (1/2/3/q): ").strip().lower()
        
        if choice == "q":
            print("\nExiting. Good bye!")
            break
            
        elif choice == "1":
            complaint = input("\nEnter complaint text: ").strip()
            if not complaint:
                print("Empty input.")
                continue
                
            res_sem = semantic_model.predict_with_uncertainty([complaint])[0]
            res_tfidf = tfidf_model.predict_with_uncertainty([complaint])[0]
            
            print("\n--- Semantic Multilingual Model ---")
            print(f"Predicted Category : {res_sem['predicted_category']}")
            print(f"Confidence Score   : {res_sem['confidence'] * 100:.1f}%")
            print(f"Routing Decision   : {res_sem['review_flag']}")
            if res_sem["reasons"]:
                print(f"Flag Reason        : {', '.join(res_sem['reasons'])}")
                
            print("\n--- Lexical TF-IDF Baseline ---")
            print(f"Predicted Category : {res_tfidf['predicted_category']}")
            print(f"Confidence Score   : {res_tfidf['confidence'] * 100:.1f}%")
            print(f"Routing Decision   : {res_tfidf['review_flag']}")
            
        elif choice == "2":
            t1 = input("\nEnter Complaint 1: ").strip()
            t2 = input("Enter Complaint 2: ").strip()
            if not t1 or not t2:
                print("Both complaints are required.")
                continue
                
            pred = similarity_engine.predict_pair(t1, t2, alpha=0.70, threshold=0.30)
            print("\n--- Similarity Breakdown ---")
            print(f"Lexical Match      : {pred['lexical_similarity'] * 100:.1f}%")
            print(f"Semantic Match     : {pred['semantic_similarity'] * 100:.1f}%")
            print(f"Hybrid Score       : {pred['hybrid_similarity'] * 100:.1f}%")
            print(f"Duplicate Verdict  : {pred['verdict']}")
            
        elif choice == "3":
            query = input("\nEnter new complaint: ").strip()
            if not query:
                continue
                
            top_matches = similarity_engine.find_top_duplicates(
                query_text=query,
                candidates=df_complaints["text"].tolist(),
                candidate_ids=df_complaints["complaint_id"].tolist(),
                candidate_categories=df_complaints["category"].tolist(),
                top_k=3,
                alpha=0.70,
                threshold=0.30
            )
            print("\n--- Top Matching Existing Complaints ---")
            for i, match in enumerate(top_matches, 1):
                flag = "POTENTIAL DUPLICATE" if match["is_potential_duplicate"] else "DIFFERENT ISSUE"
                print(f"{i}. [{match['candidate_id']}] ({match['category']}) - {flag}")
                print(f"   Match Score: {match['hybrid_similarity'] * 100:.1f}% (Semantic: {match['semantic_similarity']*100:.1f}%, Lexical: {match['lexical_similarity']*100:.1f}%)")
                print(f"   Text: {match['text']}\n")
        else:
            print("Invalid choice, please enter 1, 2, 3, or q.")

if __name__ == "__main__":
    main()
