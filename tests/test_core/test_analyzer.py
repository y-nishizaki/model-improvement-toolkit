"""Tests for ModelAnalyzer base class."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock

from model_improvement_toolkit.core.analyzer import ModelAnalyzer


class ConcreteAnalyzer(ModelAnalyzer):
    """Concrete implementation for testing."""

    def analyze(self, X, y, X_val=None, y_val=None):
        """Mock analyze method."""
        self._results = {"test": "results"}
        return self._results

    def get_model_type(self):
        """Mock get_model_type method."""
        return "test_model"


class TestModelAnalyzer:
    """Test cases for ModelAnalyzer base class."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock model."""
        return Mock()

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        X = pd.DataFrame(
            {
                "feature1": [1, 2, 3, 4, 5],
                "feature2": [5, 4, 3, 2, 1],
                "feature3": [2, 3, 4, 5, 6],
            }
        )
        y = pd.Series([0, 1, 0, 1, 0])
        return X, y

    def test_initialization(self, mock_model):
        """Test ModelAnalyzer initialization."""
        analyzer = ConcreteAnalyzer(mock_model)
        assert analyzer.model is mock_model
        assert analyzer.feature_names is None
        assert analyzer._results == {}

    def test_initialization_with_feature_names(self, mock_model):
        """Test ModelAnalyzer initialization with feature names."""
        feature_names = ["f1", "f2", "f3"]
        analyzer = ConcreteAnalyzer(mock_model, feature_names=feature_names)
        assert analyzer.feature_names == feature_names

    def test_get_results_without_analysis(self, mock_model):
        """Test get_results raises error when no analysis has been run."""
        analyzer = ConcreteAnalyzer(mock_model)
        with pytest.raises(ValueError, match="No analysis results available"):
            analyzer.get_results()

    def test_get_results_after_analysis(self, mock_model, sample_data):
        """Test get_results returns results after analysis."""
        analyzer = ConcreteAnalyzer(mock_model)
        X, y = sample_data
        analyzer.analyze(X, y)
        results = analyzer.get_results()
        assert results == {"test": "results"}

    def test_generate_report_without_analysis(self, mock_model):
        """Test generate_report raises error when no analysis has been run."""
        analyzer = ConcreteAnalyzer(mock_model)
        with pytest.raises(ValueError, match="No analysis results available"):
            analyzer.generate_report()

    def test_generate_report_unsupported_format(self, mock_model, sample_data):
        """Test generate_report raises error for unsupported format."""
        analyzer = ConcreteAnalyzer(mock_model)
        X, y = sample_data
        analyzer.analyze(X, y)
        with pytest.raises(NotImplementedError, match="Output format 'invalid' is not supported"):
            analyzer.generate_report(output_format="invalid")

    def test_validate_input_mismatched_lengths(self, mock_model):
        """Test _validate_input raises error for mismatched X and y lengths."""
        analyzer = ConcreteAnalyzer(mock_model)
        X = pd.DataFrame({"f1": [1, 2, 3]})
        y = pd.Series([0, 1])
        with pytest.raises(ValueError, match="X and y must have the same length"):
            analyzer._validate_input(X, y)

    def test_validate_input_mismatched_val_lengths(self, mock_model, sample_data):
        """Test _validate_input raises error for mismatched validation data lengths."""
        analyzer = ConcreteAnalyzer(mock_model)
        X, y = sample_data
        X_val = pd.DataFrame({"f1": [1, 2, 3]})
        y_val = pd.Series([0, 1])
        with pytest.raises(ValueError, match="X_val and y_val must have the same length"):
            analyzer._validate_input(X, y, X_val, y_val)

    def test_validate_input_different_columns(self, mock_model, sample_data):
        """Test _validate_input raises error for different columns in train/val data."""
        analyzer = ConcreteAnalyzer(mock_model)
        X, y = sample_data
        X_val = pd.DataFrame({"different_feature": [1, 2, 3, 4, 5]})
        y_val = pd.Series([0, 1, 0, 1, 0])
        with pytest.raises(ValueError, match="Training and validation data must have the same columns"):
            analyzer._validate_input(X, y, X_val, y_val)

    def test_validate_input_mismatched_feature_names(self, mock_model):
        """Test _validate_input raises error for mismatched feature names."""
        analyzer = ConcreteAnalyzer(mock_model, feature_names=["f1", "f2"])
        X = pd.DataFrame({"feature1": [1, 2, 3], "feature2": [4, 5, 6], "feature3": [7, 8, 9]})
        y = pd.Series([0, 1, 0])
        with pytest.raises(ValueError, match="Number of feature names"):
            analyzer._validate_input(X, y)

    def test_validate_input_success(self, mock_model, sample_data):
        """Test _validate_input succeeds with valid data."""
        analyzer = ConcreteAnalyzer(mock_model)
        X, y = sample_data
        # Should not raise any exception
        analyzer._validate_input(X, y)

    def test_abstract_methods(self, mock_model):
        """Test that abstract methods must be implemented."""
        with pytest.raises(TypeError):
            # Cannot instantiate abstract class
            ModelAnalyzer(mock_model)  # type: ignore