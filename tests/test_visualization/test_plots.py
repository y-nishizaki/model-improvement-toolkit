"""Tests for visualization plotting functions."""

import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from model_improvement_toolkit.visualization.plots import (
    create_summary_plots,
    plot_cumulative_importance,
    plot_feature_importance,
    plot_overfitting_diagnosis,
    plot_performance_metrics,
    set_plot_style,
)


class TestPlotFunctions:
    """Test individual plotting functions."""

    def test_set_plot_style(self):
        """Test plot style setting."""
        # Should not raise any errors
        set_plot_style("seaborn")
        set_plot_style("default")

    def test_plot_feature_importance(self):
        """Test feature importance plotting."""
        # Create test data
        importance_data = [
            {"feature": f"feature_{i}", "importance_normalized": 100 - i * 5}
            for i in range(20)
        ]

        # Test with default parameters
        fig = plot_feature_importance(importance_data)
        assert isinstance(fig, Figure)
        assert len(fig.axes) == 1

        # Test with custom parameters
        fig = plot_feature_importance(
            importance_data,
            top_n=10,
            figsize=(8, 6),
            title="Test Feature Importance"
        )
        assert isinstance(fig, Figure)

    def test_plot_cumulative_importance(self):
        """Test cumulative importance plotting."""
        # Create test data
        n_features = 50
        cumulative_data = []
        cumulative_value = 0
        for i in range(n_features):
            # Simulate decreasing importance
            increment = (1 - cumulative_value) * 0.1
            cumulative_value += increment
            cumulative_data.append({"cumulative_importance": cumulative_value})

        # Test with default parameters
        fig = plot_cumulative_importance(cumulative_data)
        assert isinstance(fig, Figure)
        assert len(fig.axes) == 1

        # Test with custom thresholds
        fig = plot_cumulative_importance(
            cumulative_data,
            thresholds=[0.6, 0.9],
            title="Test Cumulative Importance"
        )
        assert isinstance(fig, Figure)

    def test_plot_performance_metrics_classification(self):
        """Test performance metrics plotting for classification."""
        # Classification metrics
        metrics = {
            "train": {
                "accuracy": 0.95,
                "precision": 0.92,
                "recall": 0.88,
                "f1": 0.90,
                "auc": 0.96
            },
            "validation": {
                "accuracy": 0.92,
                "precision": 0.89,
                "recall": 0.85,
                "f1": 0.87,
                "auc": 0.93
            }
        }

        # Test with train and validation
        fig = plot_performance_metrics(metrics)
        assert isinstance(fig, Figure)
        assert len(fig.axes) == 2  # Two subplots

        # Test with train only
        metrics_train_only = {"train": metrics["train"]}
        fig = plot_performance_metrics(metrics_train_only)
        assert isinstance(fig, Figure)
        assert len(fig.axes) == 1  # One subplot

    def test_plot_performance_metrics_regression(self):
        """Test performance metrics plotting for regression."""
        # Regression metrics
        metrics = {
            "train": {
                "mse": 0.05,
                "rmse": 0.224,
                "mae": 0.18,
                "r2": 0.95
            },
            "validation": {
                "mse": 0.08,
                "rmse": 0.283,
                "mae": 0.22,
                "r2": 0.92
            }
        }

        fig = plot_performance_metrics(metrics)
        assert isinstance(fig, Figure)
        assert len(fig.axes) == 2

    def test_plot_overfitting_diagnosis(self):
        """Test overfitting diagnosis plotting."""
        # Test data with overfitting
        overfitting_data = {
            "has_validation": True,
            "overfitting_detected": True,
            "severity": "medium",
            "metrics": {
                "train_score": 0.95,
                "validation_score": 0.85,
                "score_gap": 0.10,
                "relative_gap": 0.105
            }
        }

        fig = plot_overfitting_diagnosis(overfitting_data)
        assert isinstance(fig, Figure)
        assert len(fig.axes) == 2

        # Test without validation data
        no_val_data = {"has_validation": False}
        fig = plot_overfitting_diagnosis(no_val_data)
        assert fig is None

    def test_plot_overfitting_diagnosis_edge_cases(self):
        """Test overfitting diagnosis with edge cases."""
        # No metrics
        data = {"has_validation": True, "metrics": {}}
        fig = plot_overfitting_diagnosis(data)
        assert fig is None

        # Missing metrics
        data = {
            "has_validation": True,
            "metrics": {"train_score": 0.9}
        }
        fig = plot_overfitting_diagnosis(data)
        assert isinstance(fig, Figure)

    def test_create_summary_plots(self):
        """Test summary plots creation."""
        # Create comprehensive test data
        analysis_results = {
            "feature_importance": {
                "importance_scores": [
                    {"feature": f"feature_{i}", "importance_normalized": 100 - i * 5}
                    for i in range(20)
                ],
                "cumulative_importance": {
                    "cumulative_importance_curve": [
                        {"cumulative_importance": i / 20}
                        for i in range(1, 21)
                    ]
                }
            },
            "performance_metrics": {
                "train": {
                    "accuracy": 0.95,
                    "precision": 0.92,
                    "recall": 0.88,
                    "f1": 0.90
                },
                "validation": {
                    "accuracy": 0.92,
                    "precision": 0.89,
                    "recall": 0.85,
                    "f1": 0.87
                }
            },
            "overfitting_analysis": {
                "has_validation": True,
                "overfitting_detected": True,
                "severity": "low",
                "metrics": {
                    "train_score": 0.95,
                    "validation_score": 0.92,
                    "score_gap": 0.03,
                    "relative_gap": 0.032
                }
            }
        }

        # Test plot creation
        plots = create_summary_plots(analysis_results)
        
        # Check all expected plots are created
        expected_plots = [
            "feature_importance",
            "cumulative_importance",
            "performance_metrics",
            "overfitting_diagnosis"
        ]
        
        for plot_name in expected_plots:
            assert plot_name in plots
            assert isinstance(plots[plot_name], Figure)

    def test_create_summary_plots_with_output_dir(self, tmp_path):
        """Test summary plots creation with output directory."""
        # Create minimal test data
        analysis_results = {
            "performance_metrics": {
                "train": {"accuracy": 0.95, "f1": 0.90}
            }
        }

        # Create plots with output directory
        output_dir = tmp_path / "plots"
        plots = create_summary_plots(analysis_results, output_dir=str(output_dir))

        # Check plots are saved
        assert output_dir.exists()
        assert (output_dir / "performance_metrics.png").exists()

    def test_create_summary_plots_missing_data(self):
        """Test summary plots with missing data."""
        # Empty results
        plots = create_summary_plots({})
        assert len(plots) == 0

        # Partial results
        analysis_results = {
            "performance_metrics": {
                "train": {"accuracy": 0.95}
            }
        }
        plots = create_summary_plots(analysis_results)
        assert "performance_metrics" in plots
        assert len(plots) == 1


class TestPlotIntegration:
    """Test plot functions with realistic data."""

    def test_realistic_feature_importance(self):
        """Test with realistic feature importance data."""
        # Simulate realistic importance scores
        np.random.seed(42)
        n_features = 100
        
        # Create power-law distributed importance
        importance_raw = np.random.pareto(1.5, n_features)
        importance_raw = np.sort(importance_raw)[::-1]
        importance_sum = importance_raw.sum()
        
        importance_data = [
            {
                "feature": f"feature_{i}",
                "importance": importance_raw[i],
                "importance_normalized": importance_raw[i] / importance_sum * 100
            }
            for i in range(n_features)
        ]

        fig = plot_feature_importance(importance_data, top_n=30)
        assert isinstance(fig, Figure)

    def test_realistic_cumulative_importance(self):
        """Test with realistic cumulative importance curve."""
        # Create realistic cumulative curve
        n_features = 100
        cumulative_data = []
        
        for i in range(n_features):
            # Logarithmic growth to 1.0
            cumulative = 1 - np.exp(-3 * (i + 1) / n_features)
            cumulative_data.append({"cumulative_importance": cumulative})

        fig = plot_cumulative_importance(
            cumulative_data,
            thresholds=[0.5, 0.8, 0.9, 0.95]
        )
        assert isinstance(fig, Figure)

    def test_edge_case_single_feature(self):
        """Test with single feature."""
        importance_data = [
            {"feature": "only_feature", "importance_normalized": 100.0}
        ]
        
        fig = plot_feature_importance(importance_data)
        assert isinstance(fig, Figure)

        cumulative_data = [{"cumulative_importance": 1.0}]
        fig = plot_cumulative_importance(cumulative_data)
        assert isinstance(fig, Figure)

    def test_edge_case_empty_metrics(self):
        """Test with empty metrics."""
        metrics = {"train": {}, "validation": {}}
        fig = plot_performance_metrics(metrics)
        assert isinstance(fig, Figure)


@pytest.mark.parametrize("style", ["seaborn", "default"])
def test_style_variations(style):
    """Test different plotting styles."""
    set_plot_style(style)
    
    # Create simple plot
    data = [
        {"feature": f"f{i}", "importance_normalized": 10 - i}
        for i in range(5)
    ]
    
    fig = plot_feature_importance(data, top_n=5)
    assert isinstance(fig, Figure)