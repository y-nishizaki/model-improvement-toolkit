"""Tests for report generation functionality."""

import base64
import io
from pathlib import Path

import matplotlib.pyplot as plt
import pytest

from model_improvement_toolkit.visualization.report_generator import ReportGenerator


class TestReportGenerator:
    """Test ReportGenerator class."""

    @pytest.fixture
    def generator(self):
        """Create a ReportGenerator instance."""
        return ReportGenerator()

    @pytest.fixture
    def sample_analysis_results(self):
        """Create sample analysis results."""
        return {
            "model_info": {
                "model_type": "lightgbm",
                "n_features": 50,
                "is_classifier": True,
                "n_iterations": 100,
            },
            "feature_importance": {
                "importance_scores": [
                    {"feature": f"feature_{i}", "importance_normalized": 10 - i * 0.5}
                    for i in range(20)
                ],
                "cumulative_importance": {
                    "cumulative_importance_curve": [
                        {"cumulative_importance": i / 20} for i in range(1, 21)
                    ],
                    "total_features": 50,
                    "features_for_90pct": 15,
                },
                "top_features": [f"feature_{i}" for i in range(20)],
                "low_importance_features": [f"feature_{i}" for i in range(40, 50)],
            },
            "performance_metrics": {
                "train": {
                    "accuracy": 0.95,
                    "precision": 0.92,
                    "recall": 0.88,
                    "f1": 0.90,
                    "auc": 0.96,
                },
                "validation": {
                    "accuracy": 0.92,
                    "precision": 0.89,
                    "recall": 0.85,
                    "f1": 0.87,
                    "auc": 0.93,
                },
            },
            "overfitting_analysis": {
                "has_validation": True,
                "overfitting_detected": True,
                "severity": "low",
                "metrics": {
                    "train_score": 0.95,
                    "validation_score": 0.92,
                    "score_gap": 0.03,
                    "relative_gap": 0.032,
                },
            },
            "suggestions": [
                {
                    "type": "feature_reduction",
                    "priority": "medium",
                    "message": "Consider removing low importance features",
                    "recommendations": [
                        "Remove features with < 1% importance",
                        "Use feature selection methods",
                    ],
                },
                {
                    "type": "overfitting",
                    "priority": "low",
                    "message": "Slight overfitting detected",
                    "recommendations": [
                        "Try increasing regularization",
                        "Add more training data",
                    ],
                },
            ],
        }

    def test_init(self, generator):
        """Test ReportGenerator initialization."""
        assert generator.style == "default"
        assert hasattr(generator, "env")

    def test_init_with_custom_style(self):
        """Test initialization with custom style."""
        gen = ReportGenerator(style="custom")
        assert gen.style == "custom"

    def test_encode_plots(self, generator):
        """Test plot encoding functionality."""
        # Create test plots
        fig1, ax1 = plt.subplots()
        ax1.plot([1, 2, 3], [1, 4, 9])

        fig2, ax2 = plt.subplots()
        ax2.bar(["A", "B", "C"], [1, 2, 3])

        plots = {"line_plot": fig1, "bar_plot": fig2}

        # Encode plots
        encoded = generator._encode_plots(plots)

        # Check encoding
        assert len(encoded) == 2
        assert "line_plot" in encoded
        assert "bar_plot" in encoded

        # Check base64 format
        for name, data in encoded.items():
            assert data.startswith("data:image/png;base64,")
            # Verify it's valid base64
            base64_str = data.split(",")[1]
            base64.b64decode(base64_str)

        # Clean up
        plt.close(fig1)
        plt.close(fig2)

    def test_prepare_template_data(self, generator, sample_analysis_results):
        """Test template data preparation."""
        plot_data = {"test_plot": "data:image/png;base64,test"}

        context = generator._prepare_template_data(sample_analysis_results, plot_data)

        # Check basic fields
        assert context["title"] == "Model Analysis Report"
        assert "generated_at" in context
        assert context["model_type"] == "lightgbm"

        # Check model info
        assert context["model_info"]["n_features"] == 50
        assert context["model_info"]["is_classifier"] is True

        # Check plots
        assert context["plots"] == plot_data

        # Check performance
        assert "train" in context["performance"]
        assert "validation" in context["performance"]

        # Check feature analysis
        assert context["feature_analysis"]["total_features"] == 50
        assert context["feature_analysis"]["features_for_90pct"] == 15
        assert context["feature_analysis"]["low_importance_count"] == 10
        assert len(context["feature_analysis"]["top_features"]) == 20

        # Check overfitting
        assert context["overfitting"]["overfitting_detected"] is True
        assert context["overfitting"]["severity"] == "low"

        # Check suggestions
        assert len(context["suggestions"]) == 2
        assert context["has_validation"] is True

    def test_render_html_template(self, generator):
        """Test HTML template rendering."""
        context = {
            "title": "Test Report",
            "generated_at": "2025-07-20 10:00:00",
            "model_type": "lightgbm",
            "model_info": {"n_features": 10, "is_classifier": True},
            "plots": {},
            "performance": {"train": {"accuracy": 0.95}},
            "feature_analysis": {
                "top_features": ["f1", "f2"],
                "low_importance_count": 5,
                "total_features": 10,
                "features_for_90pct": 3,
            },
            "overfitting": {},
            "suggestions": [],
            "has_validation": False,
        }

        html = generator._render_html_template(context)

        # Check HTML structure
        assert "<!DOCTYPE html>" in html
        assert "<html" in html
        assert "</html>" in html

        # Check content
        assert "Test Report" in html
        assert "2025-07-20 10:00:00" in html
        assert "LIGHTGBM" in html
        assert "ACCURACY" in html
        assert "0.9500" in html

    def test_generate_html_report(self, generator, sample_analysis_results, tmp_path):
        """Test HTML report generation."""
        output_path = tmp_path / "report.html"

        # Generate report
        result_path = generator.generate_html_report(
            sample_analysis_results, output_path, include_plots=True
        )

        # Check file created
        assert Path(result_path).exists()
        assert result_path == str(output_path)

        # Check content
        with open(output_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Verify HTML structure
        assert "<!DOCTYPE html>" in content
        assert "Model Analysis Report" in content
        assert "LIGHTGBM" in content

        # Check sections
        assert "Model Overview" in content
        assert "Performance Metrics" in content
        assert "Feature Analysis" in content
        assert "Overfitting Analysis" in content
        assert "Improvement Suggestions" in content

        # Check data
        assert "50" in content  # n_features
        assert "Classification" in content
        assert "0.9500" in content  # accuracy

    def test_generate_html_report_without_plots(
        self, generator, sample_analysis_results, tmp_path
    ):
        """Test HTML report generation without plots."""
        output_path = tmp_path / "report_no_plots.html"

        result_path = generator.generate_html_report(
            sample_analysis_results, output_path, include_plots=False
        )

        assert Path(result_path).exists()

        with open(output_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Should not have img tags
        assert "<img" not in content

    def test_generate_html_report_minimal_data(self, generator, tmp_path):
        """Test report generation with minimal data."""
        minimal_results = {
            "model_info": {"model_type": "unknown"},
            "suggestions": [],
        }

        output_path = tmp_path / "minimal_report.html"
        result_path = generator.generate_html_report(minimal_results, output_path)

        assert Path(result_path).exists()

        with open(output_path, "r", encoding="utf-8") as f:
            content = f.read()

        assert "Model Analysis Report" in content
        assert "UNKNOWN" in content

    def test_generate_summary_text(self, generator, sample_analysis_results):
        """Test text summary generation."""
        summary = generator.generate_summary_text(sample_analysis_results)

        # Check structure
        assert "=== MODEL ANALYSIS SUMMARY ===" in summary
        assert "Model Type: LIGHTGBM" in summary
        assert "Features: 50" in summary
        assert "Task: Classification" in summary

        # Check performance
        assert "Training Performance:" in summary
        assert "accuracy: 0.9500" in summary
        assert "Validation Performance:" in summary
        assert "accuracy: 0.9200" in summary

        # Check overfitting
        assert "Overfitting: DETECTED (Severity: low)" in summary

        # Check suggestions
        assert "Top Suggestions:" in summary
        assert "Consider removing low importance features" in summary

    def test_generate_summary_text_regression(self, generator):
        """Test text summary for regression task."""
        regression_results = {
            "model_info": {
                "model_type": "xgboost",
                "n_features": 30,
                "is_classifier": False,
            },
            "performance_metrics": {
                "train": {"mse": 0.05, "rmse": 0.224, "mae": 0.18, "r2": 0.95}
            },
            "overfitting_analysis": {"overfitting_detected": False},
            "suggestions": [],
        }

        summary = generator.generate_summary_text(regression_results)

        assert "Model Type: XGBOOST" in summary
        assert "Task: Regression" in summary
        assert "mse: 0.0500" in summary
        assert "Overfitting:" not in summary  # Not detected

    def test_generate_summary_text_minimal(self, generator):
        """Test text summary with minimal data."""
        minimal_results = {}
        summary = generator.generate_summary_text(minimal_results)

        assert "=== MODEL ANALYSIS SUMMARY ===" in summary
        assert "Model Type: UNKNOWN" in summary

    def test_plot_format_parameter(self, generator, sample_analysis_results, tmp_path):
        """Test different plot formats."""
        output_path = tmp_path / "report_svg.html"

        # Generate with SVG format
        generator.generate_html_report(
            sample_analysis_results, output_path, plot_format="svg"
        )

        with open(output_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Should have SVG data URIs
        if "<img" in content:
            assert "data:image/svg" in content

    def test_nested_directory_creation(self, generator, tmp_path):
        """Test report generation with nested directory creation."""
        nested_path = tmp_path / "reports" / "2025" / "07" / "report.html"

        minimal_results = {"model_info": {"model_type": "test"}}

        result_path = generator.generate_html_report(minimal_results, nested_path)

        assert Path(result_path).exists()
        assert nested_path.parent.exists()


class TestReportIntegration:
    """Test report generation with full pipeline."""

    def test_full_report_generation(self, tmp_path):
        """Test complete report generation flow."""
        # Create comprehensive test data
        analysis_results = {
            "model_info": {
                "model_type": "lightgbm",
                "n_features": 100,
                "is_classifier": True,
                "n_iterations": 150,
                "n_leaves": 31,
                "learning_rate": 0.1,
            },
            "feature_importance": {
                "importance_scores": [
                    {
                        "feature": f"feature_{i:03d}",
                        "importance": 1000 - i * 10,
                        "importance_normalized": (1000 - i * 10) / 10,
                    }
                    for i in range(100)
                ],
                "cumulative_importance": {
                    "cumulative_importance_curve": [
                        {"cumulative_importance": min(1.0, i * 0.02)}
                        for i in range(1, 101)
                    ],
                    "total_features": 100,
                    "features_for_90pct": 45,
                    "features_for_95pct": 65,
                },
                "top_features": [f"feature_{i:03d}" for i in range(20)],
                "low_importance_features": [f"feature_{i:03d}" for i in range(80, 100)],
            },
            "performance_metrics": {
                "train": {
                    "accuracy": 0.956,
                    "precision": 0.934,
                    "recall": 0.912,
                    "f1": 0.923,
                    "auc": 0.978,
                },
                "validation": {
                    "accuracy": 0.923,
                    "precision": 0.901,
                    "recall": 0.878,
                    "f1": 0.889,
                    "auc": 0.945,
                },
            },
            "overfitting_analysis": {
                "has_validation": True,
                "overfitting_detected": True,
                "severity": "medium",
                "metrics": {
                    "train_score": 0.956,
                    "validation_score": 0.923,
                    "score_gap": 0.033,
                    "relative_gap": 0.0345,
                },
            },
            "suggestions": [
                {
                    "type": "overfitting",
                    "priority": "high",
                    "message": "Moderate overfitting detected",
                    "recommendations": [
                        "Increase min_data_in_leaf to 40-50",
                        "Reduce num_leaves to 20-25",
                        "Add L1/L2 regularization",
                    ],
                },
                {
                    "type": "feature_reduction",
                    "priority": "medium",
                    "message": "Many low-importance features detected",
                    "recommendations": [
                        "Remove bottom 20% features",
                        "Use recursive feature elimination",
                    ],
                },
                {
                    "type": "hyperparameter_tuning",
                    "priority": "low",
                    "message": "Consider hyperparameter optimization",
                    "recommendations": ["Use Optuna or similar framework"],
                },
            ],
        }

        # Generate report
        generator = ReportGenerator()
        output_path = tmp_path / "full_report.html"

        result = generator.generate_html_report(
            analysis_results, output_path, include_plots=True
        )

        # Verify
        assert Path(result).exists()
        file_size = Path(result).stat().st_size
        assert file_size > 1000  # Should be substantial

        # Also generate text summary
        text_summary = generator.generate_summary_text(analysis_results)
        assert len(text_summary) > 200
        assert "Moderate overfitting detected" in text_summary