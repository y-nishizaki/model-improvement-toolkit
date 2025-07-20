"""Metrics calculation for model evaluation."""

from typing import Any, Dict, Optional, Union

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from ..utils.exceptions import InvalidDataError


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    task_type: str = "auto",
    average: str = "binary",
    pos_label: Optional[Union[int, str]] = None,
) -> Dict[str, float]:
    """
    Calculate metrics based on task type.

    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels or probabilities
    task_type : str, default="auto"
        Type of task ("classification", "regression", "probability", or "auto")
    average : str, default="binary"
        Averaging method for multiclass ("binary", "micro", "macro", "weighted")
    pos_label : Optional[Union[int, str]], default=None
        Positive label for binary classification

    Returns
    -------
    Dict[str, float]
        Dictionary of calculated metrics

    Raises
    ------
    InvalidDataError
        If input data is invalid
    """
    # Validate inputs
    if len(y_true) != len(y_pred):
        raise InvalidDataError(
            f"Length mismatch: y_true has {len(y_true)} samples, "
            f"y_pred has {len(y_pred)} samples"
        )

    if len(y_true) == 0:
        raise InvalidDataError("Empty arrays provided")

    # Convert to numpy arrays
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Auto-detect task type if needed
    if task_type == "auto":
        task_type = _detect_task_type(y_true, y_pred)

    if task_type == "classification":
        return _calculate_classification_metrics(y_true, y_pred, average, pos_label)
    elif task_type == "regression":
        return _calculate_regression_metrics(y_true, y_pred)
    elif task_type == "probability":
        return _calculate_probability_metrics(y_true, y_pred, pos_label)
    else:
        raise ValueError(f"Unknown task type: {task_type}")


def _detect_task_type(y_true: np.ndarray, y_pred: np.ndarray) -> str:
    """
    Detect task type from data.

    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predictions

    Returns
    -------
    str
        Detected task type
    """
    # Check if predictions are probabilities
    if y_pred.ndim == 2 or (y_pred.min() >= 0 and y_pred.max() <= 1 and 
                            not np.array_equal(y_pred, y_pred.astype(int))):
        return "probability"
    
    # Check if discrete values (classification)
    unique_true = np.unique(y_true)
    unique_pred = np.unique(y_pred)
    
    if (len(unique_true) < 20 and len(unique_pred) < 20 and
        np.array_equal(y_true, y_true.astype(int)) and
        np.array_equal(y_pred, y_pred.astype(int))):
        return "classification"
    
    return "regression"


def _calculate_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    average: str = "binary",
    pos_label: Optional[Union[int, str]] = None,
) -> Dict[str, float]:
    """
    Calculate classification metrics.

    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels
    average : str
        Averaging method
    pos_label : Optional[Union[int, str]]
        Positive label

    Returns
    -------
    Dict[str, float]
        Classification metrics
    """
    metrics = {}
    
    # Basic metrics
    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    
    # Determine if binary or multiclass
    n_classes = len(np.unique(y_true))
    
    if n_classes == 2:
        # Binary classification
        if pos_label is None:
            pos_label = 1
        
        metrics["precision"] = precision_score(y_true, y_pred, pos_label=pos_label)
        metrics["recall"] = recall_score(y_true, y_pred, pos_label=pos_label)
        metrics["f1"] = f1_score(y_true, y_pred, pos_label=pos_label)
        
        # Confusion matrix values
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics["true_positives"] = int(tp)
        metrics["true_negatives"] = int(tn)
        metrics["false_positives"] = int(fp)
        metrics["false_negatives"] = int(fn)
        
    else:
        # Multiclass classification
        metrics["precision"] = precision_score(y_true, y_pred, average=average)
        metrics["recall"] = recall_score(y_true, y_pred, average=average)
        metrics["f1"] = f1_score(y_true, y_pred, average=average)
        
        # Per-class metrics
        metrics["precision_per_class"] = precision_score(
            y_true, y_pred, average=None
        ).tolist()
        metrics["recall_per_class"] = recall_score(
            y_true, y_pred, average=None
        ).tolist()
        metrics["f1_per_class"] = f1_score(
            y_true, y_pred, average=None
        ).tolist()
    
    return metrics


def _calculate_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Calculate regression metrics.

    Parameters
    ----------
    y_true : np.ndarray
        True values
    y_pred : np.ndarray
        Predicted values

    Returns
    -------
    Dict[str, float]
        Regression metrics
    """
    metrics = {}
    
    # Basic metrics
    metrics["mse"] = mean_squared_error(y_true, y_pred)
    metrics["rmse"] = np.sqrt(metrics["mse"])
    metrics["mae"] = mean_absolute_error(y_true, y_pred)
    metrics["r2"] = r2_score(y_true, y_pred)
    
    # Additional metrics
    residuals = y_true - y_pred
    metrics["mean_residual"] = np.mean(residuals)
    metrics["std_residual"] = np.std(residuals)
    metrics["max_error"] = np.max(np.abs(residuals))
    
    # Percentage errors (avoid division by zero)
    mask = y_true != 0
    if np.any(mask):
        percentage_errors = np.abs(residuals[mask] / y_true[mask]) * 100
        metrics["mape"] = np.mean(percentage_errors)  # Mean Absolute Percentage Error
        metrics["max_percentage_error"] = np.max(percentage_errors)
    else:
        metrics["mape"] = np.nan
        metrics["max_percentage_error"] = np.nan
    
    return metrics


def _calculate_probability_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    pos_label: Optional[Union[int, str]] = None,
) -> Dict[str, float]:
    """
    Calculate metrics for probability predictions.

    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_prob : np.ndarray
        Predicted probabilities
    pos_label : Optional[Union[int, str]]
        Positive label

    Returns
    -------
    Dict[str, float]
        Probability metrics
    """
    metrics = {}
    
    # Handle multiclass probabilities
    if y_prob.ndim == 2 and y_prob.shape[1] > 2:
        # Multiclass AUC (one-vs-rest)
        try:
            from sklearn.preprocessing import label_binarize
            n_classes = y_prob.shape[1]
            y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))
            
            # Calculate AUC for each class
            auc_scores = []
            for i in range(n_classes):
                if len(np.unique(y_true_bin[:, i])) > 1:  # Check if both classes present
                    auc_scores.append(roc_auc_score(y_true_bin[:, i], y_prob[:, i]))
            
            if auc_scores:
                metrics["auc"] = np.mean(auc_scores)
                metrics["auc_per_class"] = auc_scores
        except Exception:
            metrics["auc"] = np.nan
    else:
        # Binary classification
        if y_prob.ndim == 2:
            # Probabilities for positive class
            y_prob = y_prob[:, 1]
        
        # Ensure binary labels
        unique_labels = np.unique(y_true)
        if len(unique_labels) != 2:
            raise InvalidDataError(
                f"Binary classification metrics require exactly 2 classes, "
                f"got {len(unique_labels)}"
            )
        
        # Calculate AUC
        try:
            metrics["auc"] = roc_auc_score(y_true, y_prob)
            
            # Calculate optimal threshold using Youden's J statistic
            fpr, tpr, thresholds = roc_curve(y_true, y_prob)
            j_scores = tpr - fpr
            optimal_idx = np.argmax(j_scores)
            metrics["optimal_threshold"] = float(thresholds[optimal_idx])
            
        except Exception:
            metrics["auc"] = np.nan
            metrics["optimal_threshold"] = 0.5
    
    # Log loss (cross-entropy)
    try:
        from sklearn.metrics import log_loss
        metrics["log_loss"] = log_loss(y_true, y_prob)
    except Exception:
        metrics["log_loss"] = np.nan
    
    return metrics