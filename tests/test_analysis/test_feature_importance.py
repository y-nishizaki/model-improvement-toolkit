"""Tests for feature importance analysis."""

from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from model_improvement_toolkit.analysis.feature_importance import (
    FeatureImportanceAnalyzer,
)


class TestFeatureImportanceAnalyzer:
    """Test cases for FeatureImportanceAnalyzer class."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance."""
        return FeatureImportanceAnalyzer()

    @pytest.fixture
    def mock_model(self):
        """Create mock model with feature importance."""
        model = Mock()
        model.get_feature_importance = Mock(
            return_value=np.array([0.5, 0.3, 0.15, 0.05])
        )
        return model

    @pytest.fixture
    def feature_names(self):
        """Create feature names."""
        return ["important_feature", "medium_feature", "low_feature", "minimal_feature"]

    def test_initialization(self):
        """Test analyzer initialization."""
        analyzer = FeatureImportanceAnalyzer("split")
        assert analyzer.importance_type == "split"

    def test_analyze_basic(self, analyzer, mock_model, feature_names):
        """Test basic analyze functionality."""
        results = analyzer.analyze(mock_model, feature_names)
        
        # Check result structure
        assert "importance_ranking" in results
        assert "cumulative_importance" in results
        assert "low_importance_features" in results
        assert "importance_statistics" in results
        assert "feature_groups" in results

    def test_rank_features(self, analyzer, mock_model, feature_names):
        """Test feature ranking."""
        results = analyzer.analyze(mock_model, feature_names)
        ranking = results["importance_ranking"]
        
        # Check that features are ranked by importance
        assert len(ranking) == 4
        assert ranking[0]["feature"] == "important_feature"
        assert ranking[0]["rank"] == 1
        assert ranking[0]["importance"] == 0.5
        assert ranking[0]["importance_normalized"] == 100.0
        
        assert ranking[-1]["feature"] == "minimal_feature"
        assert ranking[-1]["rank"] == 4
        assert ranking[-1]["importance"] == 0.05

    def test_cumulative_importance(self, analyzer, mock_model, feature_names):
        """Test cumulative importance calculation."""
        results = analyzer.analyze(mock_model, feature_names)
        cum_importance = results["cumulative_importance"]
        
        # Check structure
        assert "cumulative_importance_curve" in cum_importance
        assert "features_for_50pct" in cum_importance
        assert "features_for_90pct" in cum_importance
        assert "total_features" in cum_importance
        
        # Check values
        assert cum_importance["total_features"] == 4
        assert cum_importance["features_for_50pct"] == 1  # First feature has 50% importance
        assert cum_importance["features_for_90pct"] == 3  # First 3 features have 95% importance

    def test_low_importance_features(self, analyzer, mock_model, feature_names):
        """Test identification of low importance features."""
        results = analyzer.analyze(mock_model, feature_names)
        low_importance = results["low_importance_features"]
        
        # Only minimal_feature should be below 1% of max (0.5 * 0.01 = 0.005)
        assert "minimal_feature" in low_importance
        assert len(low_importance) == 1

    def test_importance_statistics(self, analyzer, mock_model, feature_names):
        """Test importance statistics calculation."""
        results = analyzer.analyze(mock_model, feature_names)
        stats = results["importance_statistics"]
        
        # Check basic statistics
        assert "mean" in stats
        assert "std" in stats
        assert "min" in stats
        assert "max" in stats
        assert "median" in stats
        assert "cv" in stats
        assert "gini" in stats
        
        # Check values
        assert stats["max"] == 0.5
        assert stats["min"] == 0.05
        assert stats["mean"] == 0.25  # (0.5 + 0.3 + 0.15 + 0.05) / 4
        assert 0 <= stats["gini"] <= 1

    def test_feature_groups(self, analyzer, mock_model, feature_names):
        """Test feature grouping by importance."""
        results = analyzer.analyze(mock_model, feature_names)
        groups = results["feature_groups"]
        
        # Check that all groups exist
        assert "critical_importance" in groups
        assert "high_importance" in groups
        assert "medium_importance" in groups
        assert "low_importance" in groups
        
        # Check that all features are assigned
        all_grouped_features = []
        for group_features in groups.values():
            all_grouped_features.extend(group_features)
        
        assert len(all_grouped_features) == 4
        assert set(all_grouped_features) == set(feature_names)

    def test_suggest_feature_selection(self, analyzer, mock_model, feature_names):
        """Test feature selection suggestions."""
        # First analyze
        analysis_results = analyzer.analyze(mock_model, feature_names)
        
        # Then suggest feature selection
        selection = analyzer.suggest_feature_selection(
            analysis_results, threshold=0.9, min_features=2
        )
        
        assert "selected_features" in selection
        assert "n_features_selected" in selection
        assert "n_features_removed" in selection
        assert "reduction_percentage" in selection
        assert "removed_features" in selection
        
        # Should select top 3 features for 90% threshold
        assert selection["n_features_selected"] == 3
        assert selection["n_features_removed"] == 1
        assert "important_feature" in selection["selected_features"]
        assert "minimal_feature" in selection["removed_features"]

    def test_suggest_feature_selection_min_features(self, analyzer, mock_model, feature_names):
        """Test feature selection with minimum features constraint."""
        analysis_results = analyzer.analyze(mock_model, feature_names)
        
        # Set high threshold but require minimum 3 features
        selection = analyzer.suggest_feature_selection(
            analysis_results, threshold=0.5, min_features=3
        )
        
        # Should respect minimum features even if threshold is met earlier
        assert selection["n_features_selected"] >= 3

    def test_get_importance_scores_sklearn(self, analyzer, feature_names):
        """Test getting importance scores from sklearn-style model."""
        model = Mock()
        model.feature_importances_ = np.array([0.4, 0.3, 0.2, 0.1])
        
        scores = analyzer._get_importance_scores(model, "gain")
        
        assert np.array_equal(scores, [0.4, 0.3, 0.2, 0.1])

    def test_get_importance_scores_custom_method(self, analyzer, feature_names):
        """Test getting importance scores from custom method."""
        model = Mock()
        model.feature_importance = Mock(
            return_value=np.array([0.3, 0.3, 0.2, 0.2])
        )
        
        scores = analyzer._get_importance_scores(model, "split")
        
        model.feature_importance.assert_called_once_with(importance_type="split")
        assert np.array_equal(scores, [0.3, 0.3, 0.2, 0.2])

    def test_get_importance_scores_no_method(self, analyzer, feature_names):
        """Test error when model has no importance method."""
        model = Mock()
        # Remove all importance-related attributes
        if hasattr(model, "feature_importances_"):
            delattr(model, "feature_importances_")
        
        with pytest.raises(AttributeError, match="doesn't have feature importance"):
            analyzer._get_importance_scores(model, "gain")

    def test_gini_coefficient_calculation(self, analyzer):
        """Test Gini coefficient calculation."""
        # Perfect equality (all values same)
        equal_values = np.array([1, 1, 1, 1])
        gini_equal = analyzer._calculate_gini_coefficient(equal_values)
        assert abs(gini_equal) < 1e-10  # Should be very close to 0
        
        # Perfect inequality (one value, rest zeros)
        unequal_values = np.array([0, 0, 0, 1])
        gini_unequal = analyzer._calculate_gini_coefficient(unequal_values)
        assert gini_unequal > 0.7  # Should be high
        
        # Empty array
        empty_values = np.array([])
        gini_empty = analyzer._calculate_gini_coefficient(empty_values)
        assert gini_empty == 0

    def test_zero_importance_handling(self, analyzer, feature_names):
        """Test handling of all-zero importance values."""
        model = Mock()
        model.get_feature_importance = Mock(
            return_value=np.array([0.0, 0.0, 0.0, 0.0])
        )
        
        results = analyzer.analyze(model, feature_names)
        
        # Should handle zero importance gracefully
        ranking = results["importance_ranking"]
        assert all(item["importance_normalized"] == 0 for item in ranking)
        
        # No low importance features when all are zero
        assert len(results["low_importance_features"]) == 0

    def test_analyze_with_override_importance_type(self, analyzer, mock_model, feature_names):
        """Test analyze with overridden importance type."""
        # Mock different return for split importance
        mock_model.get_feature_importance.side_effect = lambda imp_type: (
            np.array([0.2, 0.4, 0.3, 0.1]) if imp_type == "split"
            else np.array([0.5, 0.3, 0.15, 0.05])
        )
        
        results = analyzer.analyze(mock_model, feature_names, importance_type="split")
        
        # Should use split importance, so medium_feature should be top
        ranking = results["importance_ranking"]
        assert ranking[0]["feature"] == "medium_feature"
        assert ranking[0]["importance"] == 0.4

    def test_single_feature(self, analyzer):
        """Test with single feature."""
        model = Mock()
        model.get_feature_importance = Mock(return_value=np.array([1.0]))
        feature_names = ["only_feature"]
        
        results = analyzer.analyze(model, feature_names)
        
        # Should work with single feature
        assert len(results["importance_ranking"]) == 1
        assert results["importance_ranking"][0]["feature"] == "only_feature"
        assert results["cumulative_importance"]["features_for_90pct"] == 1