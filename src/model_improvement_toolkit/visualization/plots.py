"""Basic plotting functions for model analysis visualization."""

from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure


def set_plot_style(style: str = "seaborn") -> None:
    """
    Set the plotting style.

    Parameters
    ----------
    style : str, default="seaborn"
        Style to use for plots
    """
    if style == "seaborn":
        sns.set_theme()
    else:
        plt.style.use(style)


def plot_feature_importance(
    importance_data: List[Dict[str, Any]],
    top_n: int = 20,
    figsize: Tuple[int, int] = (10, 6),
    title: Optional[str] = None,
) -> Figure:
    """
    Plot feature importance as a horizontal bar chart.

    Parameters
    ----------
    importance_data : List[Dict[str, Any]]
        List of dictionaries with 'feature' and 'importance' keys
    top_n : int, default=20
        Number of top features to display
    figsize : Tuple[int, int], default=(10, 6)
        Figure size
    title : Optional[str], default=None
        Plot title

    Returns
    -------
    Figure
        Matplotlib figure object
    """
    # Convert to DataFrame for easier handling
    df = pd.DataFrame(importance_data)
    
    # Sort and take top N
    df_top = df.nlargest(top_n, "importance_normalized")
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create horizontal bar plot
    bars = ax.barh(
        df_top["feature"],
        df_top["importance_normalized"],
        color="steelblue"
    )
    
    # Add value labels on bars
    for i, (idx, row) in enumerate(df_top.iterrows()):
        ax.text(
            row["importance_normalized"] + 1,
            i,
            f'{row["importance_normalized"]:.1f}',
            va="center",
            fontsize=9
        )
    
    # Customize plot
    ax.set_xlabel("Relative Importance (%)")
    ax.set_ylabel("Features")
    ax.set_title(title or f"Top {top_n} Feature Importance")
    ax.set_xlim(0, 105)
    
    # Tight layout
    plt.tight_layout()
    
    return fig


def plot_cumulative_importance(
    cumulative_data: List[Dict[str, Any]],
    thresholds: List[float] = [0.5, 0.8, 0.95],
    figsize: Tuple[int, int] = (10, 6),
    title: Optional[str] = None,
) -> Figure:
    """
    Plot cumulative feature importance curve.

    Parameters
    ----------
    cumulative_data : List[Dict[str, Any]]
        List with 'cumulative_importance' values
    thresholds : List[float], default=[0.5, 0.8, 0.95]
        Thresholds to highlight
    figsize : Tuple[int, int], default=(10, 6)
        Figure size
    title : Optional[str], default=None
        Plot title

    Returns
    -------
    Figure
        Matplotlib figure object
    """
    # Extract cumulative importance values
    cumulative_values = [item["cumulative_importance"] for item in cumulative_data]
    n_features = len(cumulative_values)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot cumulative importance
    ax.plot(range(1, n_features + 1), cumulative_values, 
            color="darkblue", linewidth=2)
    
    # Add threshold lines
    colors = ["red", "orange", "green"]
    for threshold, color in zip(thresholds, colors):
        # Find number of features for threshold
        n_threshold = next(
            (i + 1 for i, cum in enumerate(cumulative_values) if cum >= threshold),
            n_features
        )
        ax.axhline(y=threshold, color=color, linestyle="--", alpha=0.7,
                  label=f"{threshold*100:.0f}% ({n_threshold} features)")
        ax.axvline(x=n_threshold, color=color, linestyle="--", alpha=0.5)
    
    # Customize plot
    ax.set_xlabel("Number of Features")
    ax.set_ylabel("Cumulative Importance")
    ax.set_title(title or "Cumulative Feature Importance")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, n_features)
    ax.set_ylim(0, 1.05)
    
    # Tight layout
    plt.tight_layout()
    
    return fig


def plot_performance_metrics(
    metrics: Dict[str, Dict[str, float]],
    figsize: Tuple[int, int] = (12, 6),
    title: Optional[str] = None,
) -> Figure:
    """
    Plot performance metrics comparison.

    Parameters
    ----------
    metrics : Dict[str, Dict[str, float]]
        Metrics dictionary with 'train' and optionally 'validation' keys
    figsize : Tuple[int, int], default=(12, 6)
        Figure size
    title : Optional[str], default=None
        Plot title

    Returns
    -------
    Figure
        Matplotlib figure object
    """
    # Prepare data
    has_validation = "validation" in metrics
    
    if has_validation:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(figsize[0]//2, figsize[1]))
    
    # Classification metrics
    classification_metrics = ["accuracy", "precision", "recall", "f1", "auc"]
    regression_metrics = ["mse", "rmse", "mae", "r2"]
    
    # Determine which metrics to plot
    train_metrics = metrics.get("train", {})
    available_metrics = list(train_metrics.keys())
    
    # Filter metrics
    if any(m in available_metrics for m in classification_metrics):
        plot_metrics = [m for m in classification_metrics if m in available_metrics]
        metric_type = "Classification"
    else:
        plot_metrics = [m for m in regression_metrics if m in available_metrics]
        metric_type = "Regression"
    
    # Plot training metrics
    if plot_metrics:
        values = [train_metrics.get(m, 0) for m in plot_metrics]
        bars1 = ax1.bar(plot_metrics, values, color="steelblue", label="Train")
        
        # Add value labels
        for bar, val in zip(bars1, values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9)
        
        ax1.set_ylabel("Score")
        ax1.set_title(f"Training {metric_type} Metrics")
        ax1.set_ylim(0, max(1.1, max(values) * 1.1))
        ax1.tick_params(axis='x', rotation=45)
    
    # Plot validation comparison if available
    if has_validation:
        val_metrics = metrics.get("validation", {})
        
        if plot_metrics:
            train_vals = [train_metrics.get(m, 0) for m in plot_metrics]
            val_vals = [val_metrics.get(m, 0) for m in plot_metrics]
            
            x = np.arange(len(plot_metrics))
            width = 0.35
            
            bars1 = ax2.bar(x - width/2, train_vals, width, 
                           label='Train', color='steelblue')
            bars2 = ax2.bar(x + width/2, val_vals, width, 
                           label='Validation', color='lightcoral')
            
            # Add value labels
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                            f'{height:.3f}', ha='center', va='bottom', fontsize=8)
            
            ax2.set_ylabel("Score")
            ax2.set_title("Train vs Validation Comparison")
            ax2.set_xticks(x)
            ax2.set_xticklabels(plot_metrics)
            ax2.legend()
            ax2.set_ylim(0, max(1.1, max(max(train_vals), max(val_vals)) * 1.1))
            ax2.tick_params(axis='x', rotation=45)
    
    plt.suptitle(title or f"{metric_type} Performance Metrics")
    plt.tight_layout()
    
    return fig


def plot_confusion_matrix(
    confusion_matrix: np.ndarray,
    class_names: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (8, 6),
    title: Optional[str] = None,
    cmap: str = "Blues",
) -> Figure:
    """
    Plot confusion matrix heatmap.

    Parameters
    ----------
    confusion_matrix : np.ndarray
        Confusion matrix array
    class_names : Optional[List[str]], default=None
        Names for classes
    figsize : Tuple[int, int], default=(8, 6)
        Figure size
    title : Optional[str], default=None
        Plot title
    cmap : str, default="Blues"
        Colormap to use

    Returns
    -------
    Figure
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    sns.heatmap(
        confusion_matrix,
        annot=True,
        fmt="d",
        cmap=cmap,
        square=True,
        cbar_kws={"label": "Count"},
        ax=ax
    )
    
    # Set labels
    if class_names:
        ax.set_xticklabels(class_names)
        ax.set_yticklabels(class_names)
    
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title or "Confusion Matrix")
    
    plt.tight_layout()
    
    return fig


def plot_overfitting_diagnosis(
    overfitting_data: Dict[str, Any],
    figsize: Tuple[int, int] = (10, 6),
    title: Optional[str] = None,
) -> Optional[Figure]:
    """
    Plot overfitting diagnosis visualization.

    Parameters
    ----------
    overfitting_data : Dict[str, Any]
        Overfitting analysis data
    figsize : Tuple[int, int], default=(10, 6)
        Figure size
    title : Optional[str], default=None
        Plot title

    Returns
    -------
    Optional[Figure]
        Matplotlib figure object or None if no validation data
    """
    if not overfitting_data.get("has_validation"):
        return None
    
    metrics = overfitting_data.get("metrics", {})
    if not metrics:
        return None
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot train vs validation scores
    scores = ["train_score", "validation_score"]
    values = [metrics.get(s, 0) for s in scores]
    colors = ["steelblue", "lightcoral"]
    
    bars = ax1.bar(["Training", "Validation"], values, color=colors)
    
    # Add value labels
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{val:.3f}', ha='center', va='bottom')
    
    ax1.set_ylabel("Score")
    ax1.set_title("Train vs Validation Performance")
    ax1.set_ylim(0, 1.1)
    
    # Plot overfitting metrics
    gap_data = {
        "Score Gap": metrics.get("score_gap", 0),
        "Relative Gap": metrics.get("relative_gap", 0),
    }
    
    bars2 = ax2.bar(gap_data.keys(), gap_data.values(), 
                    color=["orange", "red"])
    
    # Add value labels
    for bar, (key, val) in zip(bars2, gap_data.items()):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{val:.3f}', ha='center', va='bottom')
    
    ax2.set_ylabel("Value")
    ax2.set_title("Overfitting Indicators")
    
    # Add severity indicator
    severity = overfitting_data.get("severity", "unknown")
    fig.text(0.5, 0.02, f"Overfitting Severity: {severity.upper()}", 
            ha='center', fontsize=12, weight='bold',
            color="red" if severity == "high" else "orange" if severity == "medium" else "green")
    
    plt.suptitle(title or "Overfitting Diagnosis")
    plt.tight_layout()
    
    return fig


def create_summary_plots(
    analysis_results: Dict[str, Any],
    output_dir: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8),
) -> Dict[str, Figure]:
    """
    Create all summary plots from analysis results.

    Parameters
    ----------
    analysis_results : Dict[str, Any]
        Complete analysis results
    output_dir : Optional[str], default=None
        Directory to save plots
    figsize : Tuple[int, int], default=(12, 8)
        Default figure size

    Returns
    -------
    Dict[str, Figure]
        Dictionary of plot names to figure objects
    """
    plots = {}
    
    # Set style
    set_plot_style()
    
    # Feature importance plot
    if "feature_importance" in analysis_results:
        importance_data = analysis_results["feature_importance"].get(
            "importance_scores", []
        )
        if importance_data:
            plots["feature_importance"] = plot_feature_importance(
                importance_data, top_n=20, figsize=figsize
            )
    
    # Cumulative importance plot
    if "feature_importance" in analysis_results:
        cumulative_data = analysis_results["feature_importance"].get(
            "cumulative_importance", {}
        ).get("cumulative_importance_curve", [])
        if cumulative_data:
            plots["cumulative_importance"] = plot_cumulative_importance(
                cumulative_data, figsize=figsize
            )
    
    # Performance metrics plot
    if "performance_metrics" in analysis_results:
        plots["performance_metrics"] = plot_performance_metrics(
            analysis_results["performance_metrics"], figsize=figsize
        )
    
    # Overfitting diagnosis plot
    if "overfitting_analysis" in analysis_results:
        overfitting_plot = plot_overfitting_diagnosis(
            analysis_results["overfitting_analysis"], figsize=figsize
        )
        if overfitting_plot:
            plots["overfitting_diagnosis"] = overfitting_plot
    
    # Save plots if output directory specified
    if output_dir:
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        for name, fig in plots.items():
            fig.savefig(
                os.path.join(output_dir, f"{name}.png"),
                dpi=100,
                bbox_inches="tight"
            )
    
    return plots