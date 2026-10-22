"""
CivicLens Visualization Module (Phase 5)
Plotting functions for confusion matrices, robustness degradation, and similarity distributions.
"""
import os
from typing import List, Optional, Any, Tuple, Union
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# Clean styling configuration
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"

def plot_confusion_matrix(
    cm: Union[np.ndarray, List[List[int]]],
    labels: List[str],
    title: str = "Confusion Matrix",
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plots and formats a confusion matrix heatmap."""
    cm_arr = np.array(cm)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm_arr,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
        cbar=False
    )
    ax.set_title(title, fontsize=13, weight="bold", pad=12)
    ax.set_ylabel("True Category", fontsize=11, weight="bold")
    ax.set_xlabel("Predicted Category", fontsize=11, weight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig

def plot_robustness_comparison(
    comparison_data: List[dict],
    title: str = "Linguistic Robustness: English vs Code-Mixed Performance",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plots a grouped bar chart comparing performance on Standard English vs Code-Mixed inputs across models.
    """
    df = pd.DataFrame(comparison_data)
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Reshape for seaborn barplot
    melted = df.melt(id_vars=["Model"], value_vars=["English_F1", "CodeMixed_F1"],
                     var_name="Condition", value_name="Macro_F1")
    melted["Condition"] = melted["Condition"].map({
        "English_F1": "Standard English",
        "CodeMixed_F1": "Code-Mixed Indian"
    })
    
    sns.barplot(data=melted, x="Model", y="Macro_F1", hue="Condition", ax=ax, palette=["#3366CC", "#FF9900"])
    ax.set_title(title, fontsize=12, weight="bold", pad=12)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Macro F1-Score", fontsize=11, weight="bold")
    ax.set_xlabel("Model Architecture", fontsize=11, weight="bold")
    ax.legend(title="Input Condition", loc="lower right")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig

def plot_similarity_score_distributions(
    dup_scores: List[float],
    non_dup_scores: List[float],
    optimal_threshold: float = 0.50,
    title: str = "Pairwise Similarity Distribution and Separation",
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plots score distributions of duplicate pairs vs distinct incident pairs."""
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.kdeplot(dup_scores, label="Duplicate Incidents", fill=True, color="#2CA02C", alpha=0.35, ax=ax)
    sns.kdeplot(non_dup_scores, label="Distinct Incidents", fill=True, color="#D62728", alpha=0.35, ax=ax)
    
    ax.axvline(optimal_threshold, color="#333333", linestyle="--", linewidth=1.5,
               label=f"Optimal Decision Boundary (tau = {optimal_threshold:.2f})")
    ax.set_title(title, fontsize=12, weight="bold", pad=12)
    ax.set_xlabel("Similarity Score", fontsize=11, weight="bold")
    ax.set_ylabel("Density", fontsize=11, weight="bold")
    ax.set_xlim(0, 1.0)
    ax.legend(loc="upper left")
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig
