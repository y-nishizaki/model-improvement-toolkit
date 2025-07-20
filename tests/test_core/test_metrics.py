"""Tests for metrics calculation."""

import numpy as np
import pytest

from model_improvement_toolkit.core.metrics import (
    calculate_metrics,
    _detect_task_type,
    _calculate_classification_metrics,
    _calculate_regression_metrics,
    _calculate_probability_metrics,
)
from model_improvement_toolkit.utils.exceptions import InvalidDataError


class TestCalculateMetrics:
    """Test cases for calculate_metrics function."""

    def test_auto_detect_classification(self):
        """Test auto-detection of classification task."""
        y_true = np.array([0, 1, 0, 1, 0])
        y_pred = np.array([0, 1, 1, 1, 0])
        
        metrics = calculate_metrics(y_true, y_pred, task_type="auto")
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics

    def test_auto_detect_regression(self):
        """Test auto-detection of regression task."""
        y_true = np.array([1.5, 2.7, 3.2, 4.1, 5.8])
        y_pred = np.array([1.6, 2.5, 3.3, 4.0, 5.9])
        
        metrics = calculate_metrics(y_true, y_pred, task_type="auto")
        assert "mse" in metrics
        assert "rmse" in metrics
        assert "mae" in metrics
        assert "r2" in metrics

    def test_auto_detect_probability(self):
        """Test auto-detection of probability predictions."""
        y_true = np.array([0, 1, 0, 1, 0])
        y_pred = np.array([0.1, 0.9, 0.3, 0.8, 0.2])
        
        metrics = calculate_metrics(y_true, y_pred, task_type="auto")
        assert "auc" in metrics
        assert "log_loss" in metrics

    def test_length_mismatch(self):
        """Test error on length mismatch."""
        y_true = np.array([0, 1, 0])
        y_pred = np.array([0, 1])
        
        with pytest.raises(InvalidDataError, match="Length mismatch"):
            calculate_metrics(y_true, y_pred)

    def test_empty_arrays(self):
        """Test error on empty arrays."""
        y_true = np.array([])
        y_pred = np.array([])
        
        with pytest.raises(InvalidDataError, match="Empty arrays"):
            calculate_metrics(y_true, y_pred)

    def test_unknown_task_type(self):
        """Test error on unknown task type."""
        y_true = np.array([0, 1, 0])
        y_pred = np.array([0, 1, 0])
        
        with pytest.raises(ValueError, match="Unknown task type"):
            calculate_metrics(y_true, y_pred, task_type="unknown")


class TestDetectTaskType:
    """Test cases for _detect_task_type function."""

    def test_detect_classification(self):
        """Test detection of classification task."""
        y_true = np.array([0, 1, 2, 1, 0, 2])
        y_pred = np.array([0, 1, 2, 2, 0, 1])
        
        assert _detect_task_type(y_true, y_pred) == "classification"

    def test_detect_regression(self):
        """Test detection of regression task."""
        y_true = np.array([1.5, 2.7, 3.2, 4.1, 5.8])
        y_pred = np.array([1.6, 2.5, 3.3, 4.0, 5.9])
        
        assert _detect_task_type(y_true, y_pred) == "regression"

    def test_detect_probability_1d(self):
        """Test detection of 1D probability predictions."""
        y_true = np.array([0, 1, 0, 1, 0])
        y_pred = np.array([0.1, 0.9, 0.3, 0.8, 0.2])
        
        assert _detect_task_type(y_true, y_pred) == "probability"

    def test_detect_probability_2d(self):
        """Test detection of 2D probability predictions."""
        y_true = np.array([0, 1, 0, 1, 0])
        y_pred = np.array([[0.9, 0.1], [0.1, 0.9], [0.7, 0.3], [0.2, 0.8], [0.8, 0.2]])
        
        assert _detect_task_type(y_true, y_pred) == "probability"


class TestClassificationMetrics:
    """Test cases for classification metrics."""

    def test_binary_classification_metrics(self):
        """Test binary classification metrics."""
        y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1, 1, 1, 0, 0])
        
        metrics = _calculate_classification_metrics(y_true, y_pred)
        
        assert metrics["accuracy"] == 0.75  # 6/8 correct
        assert metrics["precision"] == 0.75  # 3/(3+1)
        assert metrics["recall"] == 0.75  # 3/(3+1)
        assert metrics["f1"] == 0.75
        assert metrics["true_positives"] == 3
        assert metrics["true_negatives"] == 3
        assert metrics["false_positives"] == 1
        assert metrics["false_negatives"] == 1

    def test_multiclass_classification_metrics(self):
        """Test multiclass classification metrics."""
        y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 2, 2, 1, 1, 2])
        
        metrics = _calculate_classification_metrics(y_true, y_pred, average="macro")
        
        assert metrics["accuracy"] == pytest.approx(0.7778, rel=1e-3)
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert "precision_per_class" in metrics
        assert len(metrics["precision_per_class"]) == 3

    def test_binary_with_custom_pos_label(self):
        """Test binary classification with custom positive label."""
        y_true = np.array([0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 1, 0, 0])
        
        metrics = _calculate_classification_metrics(y_true, y_pred, pos_label=0)
        
        # When pos_label=0, we're measuring performance for class 0
        assert metrics["precision"] == 1.0  # 2/(2+0) for class 0
        assert metrics["recall"] == pytest.approx(0.667, rel=1e-2)  # 2/3 for class 0


class TestRegressionMetrics:
    """Test cases for regression metrics."""

    def test_perfect_predictions(self):
        """Test metrics with perfect predictions."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        metrics = _calculate_regression_metrics(y_true, y_pred)
        
        assert metrics["mse"] == 0.0
        assert metrics["rmse"] == 0.0
        assert metrics["mae"] == 0.0
        assert metrics["r2"] == 1.0
        assert metrics["mean_residual"] == 0.0
        assert metrics["max_error"] == 0.0

    def test_regression_with_errors(self):
        """Test metrics with prediction errors."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 1.9, 3.2, 3.8, 5.1])
        
        metrics = _calculate_regression_metrics(y_true, y_pred)
        
        assert metrics["mse"] == pytest.approx(0.02, rel=1e-2)
        assert metrics["rmse"] == pytest.approx(0.1414, rel=1e-2)
        assert metrics["mae"] == pytest.approx(0.14, rel=1e-2)
        assert metrics["r2"] > 0.95
        assert metrics["mape"] < 10  # Less than 10% error

    def test_regression_with_zero_values(self):
        """Test MAPE calculation with zero values."""
        y_true = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([0.1, 1.1, 2.1, 3.1, 4.1])
        
        metrics = _calculate_regression_metrics(y_true, y_pred)
        
        assert "mape" in metrics
        assert not np.isnan(metrics["mape"])  # Should handle zeros properly

    def test_regression_all_zeros(self):
        """Test metrics when all true values are zero."""
        y_true = np.array([0.0, 0.0, 0.0, 0.0])
        y_pred = np.array([0.1, 0.2, 0.1, 0.2])
        
        metrics = _calculate_regression_metrics(y_true, y_pred)
        
        assert np.isnan(metrics["mape"])  # Can't calculate percentage error


class TestProbabilityMetrics:
    """Test cases for probability metrics."""

    def test_binary_probability_metrics(self):
        """Test binary probability metrics."""
        y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        y_prob = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6])
        
        metrics = _calculate_probability_metrics(y_true, y_prob)
        
        assert metrics["auc"] == 1.0  # Perfect separation
        assert "optimal_threshold" in metrics
        assert "log_loss" in metrics
        assert metrics["log_loss"] > 0

    def test_binary_probability_2d(self):
        """Test binary probability metrics with 2D input."""
        y_true = np.array([0, 1, 0, 1, 0])
        y_prob = np.array([[0.9, 0.1], [0.1, 0.9], [0.8, 0.2], [0.2, 0.8], [0.7, 0.3]])
        
        metrics = _calculate_probability_metrics(y_true, y_prob)
        
        assert metrics["auc"] == 1.0
        assert "optimal_threshold" in metrics

    def test_multiclass_probability_metrics(self):
        """Test multiclass probability metrics."""
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_prob = np.array([
            [0.9, 0.05, 0.05],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
            [0.8, 0.1, 0.1],
            [0.2, 0.7, 0.1],
            [0.1, 0.2, 0.7]
        ])
        
        metrics = _calculate_probability_metrics(y_true, y_prob)
        
        assert "auc" in metrics
        assert "auc_per_class" in metrics
        assert len(metrics["auc_per_class"]) == 3
        assert "log_loss" in metrics

    def test_invalid_binary_labels(self):
        """Test error with non-binary labels for binary probability."""
        y_true = np.array([0, 1, 2, 3])  # More than 2 classes
        y_prob = np.array([0.1, 0.9, 0.5, 0.7])
        
        with pytest.raises(InvalidDataError, match="exactly 2 classes"):
            _calculate_probability_metrics(y_true, y_prob)

    def test_edge_cases(self):
        """Test edge cases for probability metrics."""
        # All same class
        y_true = np.array([1, 1, 1, 1])
        y_prob = np.array([0.6, 0.7, 0.8, 0.9])
        
        metrics = _calculate_probability_metrics(y_true, y_prob)
        
        # AUC should be NaN when only one class present
        assert np.isnan(metrics["auc"])