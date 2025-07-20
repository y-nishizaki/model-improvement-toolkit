"""Tests for LightGBM wrapper and analyzer."""

from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from model_improvement_toolkit.models.lightgbm import (
    LightGBMAnalyzer,
    LightGBMWrapper,
)
from model_improvement_toolkit.utils.exceptions import UnsupportedModelError


class TestLightGBMWrapper:
    """Test cases for LightGBMWrapper class."""

    @pytest.fixture
    def mock_lgbm_sklearn_model(self):
        """Create mock LightGBM sklearn model."""
        model = Mock()
        model.__class__.__name__ = "LGBMClassifier"
        model.__class__.__module__ = "lightgbm.sklearn"
        model.n_features_ = 5
        model.n_classes_ = 2
        model.objective_ = "binary"
        model.feature_name_ = ["f1", "f2", "f3", "f4", "f5"]
        model.feature_importances_ = np.array([0.3, 0.25, 0.2, 0.15, 0.1])
        
        # Mock methods
        model.predict = Mock(return_value=np.array([0, 1, 0, 1, 0]))
        model.predict_proba = Mock(return_value=np.array([
            [0.7, 0.3], [0.2, 0.8], [0.6, 0.4], [0.3, 0.7], [0.8, 0.2]
        ]))
        model.get_params = Mock(return_value={
            "boosting_type": "gbdt",
            "n_estimators": 100,
            "learning_rate": 0.1,
            "num_leaves": 31,
        })
        
        return model

    @pytest.fixture
    def mock_lgbm_booster(self):
        """Create mock LightGBM Booster."""
        model = Mock()
        model.__class__.__name__ = "Booster"
        model.__class__.__module__ = "lightgbm.basic"
        model.params = {
            "objective": "regression",
            "boosting_type": "gbdt",
            "learning_rate": 0.1,
        }
        
        # Mock methods
        model.predict = Mock(return_value=np.array([1.5, 2.3, 3.1, 2.8, 1.9]))
        model.feature_importance = Mock(return_value=np.array([0.4, 0.3, 0.2, 0.1]))
        model.feature_name = Mock(return_value=["feat1", "feat2", "feat3", "feat4"])
        model.num_feature = Mock(return_value=4)
        model.num_trees = Mock(return_value=100)
        model.current_iteration = Mock(return_value=100)
        
        return model

    def test_wrapper_initialization_sklearn(self, mock_lgbm_sklearn_model):
        """Test wrapper initialization with sklearn model."""
        wrapper = LightGBMWrapper(mock_lgbm_sklearn_model)
        assert wrapper.model is mock_lgbm_sklearn_model

    def test_wrapper_initialization_booster(self, mock_lgbm_booster):
        """Test wrapper initialization with booster."""
        wrapper = LightGBMWrapper(mock_lgbm_booster)
        assert wrapper.model is mock_lgbm_booster

    def test_wrapper_initialization_invalid_model(self):
        """Test wrapper initialization with invalid model."""
        model = Mock()
        model.__class__.__name__ = "RandomForestClassifier"
        model.__class__.__module__ = "sklearn.ensemble"
        
        with pytest.raises(UnsupportedModelError):
            LightGBMWrapper(model)

    def test_predict_sklearn(self, mock_lgbm_sklearn_model):
        """Test predict method with sklearn model."""
        wrapper = LightGBMWrapper(mock_lgbm_sklearn_model)
        X = pd.DataFrame(np.random.randn(5, 5))
        
        predictions = wrapper.predict(X)
        
        mock_lgbm_sklearn_model.predict.assert_called_once_with(X)
        assert len(predictions) == 5

    def test_predict_booster(self, mock_lgbm_booster):
        """Test predict method with booster."""
        wrapper = LightGBMWrapper(mock_lgbm_booster)
        X = pd.DataFrame(np.random.randn(5, 4))
        
        predictions = wrapper.predict(X)
        
        mock_lgbm_booster.predict.assert_called_once_with(X.values)
        assert len(predictions) == 5

    def test_predict_proba_sklearn(self, mock_lgbm_sklearn_model):
        """Test predict_proba with sklearn classifier."""
        wrapper = LightGBMWrapper(mock_lgbm_sklearn_model)
        X = pd.DataFrame(np.random.randn(5, 5))
        
        probas = wrapper.predict_proba(X)
        
        mock_lgbm_sklearn_model.predict_proba.assert_called_once_with(X)
        assert probas.shape == (5, 2)

    def test_predict_proba_regression_error(self, mock_lgbm_booster):
        """Test predict_proba raises error for regression."""
        wrapper = LightGBMWrapper(mock_lgbm_booster)
        X = pd.DataFrame(np.random.randn(5, 4))
        
        with pytest.raises(NotImplementedError, match="not supported for regression"):
            wrapper.predict_proba(X)

    def test_get_params_sklearn(self, mock_lgbm_sklearn_model):
        """Test get_params with sklearn model."""
        wrapper = LightGBMWrapper(mock_lgbm_sklearn_model)
        
        params = wrapper.get_params()
        
        mock_lgbm_sklearn_model.get_params.assert_called_once()
        assert isinstance(params, dict)

    def test_get_params_booster(self, mock_lgbm_booster):
        """Test get_params with booster."""
        wrapper = LightGBMWrapper(mock_lgbm_booster)
        
        params = wrapper.get_params()
        
        assert params == mock_lgbm_booster.params

    def test_get_feature_importance_sklearn(self, mock_lgbm_sklearn_model):
        """Test get_feature_importance with sklearn model."""
        wrapper = LightGBMWrapper(mock_lgbm_sklearn_model)
        
        importance = wrapper.get_feature_importance()
        
        assert len(importance) == 5
        assert np.array_equal(importance, mock_lgbm_sklearn_model.feature_importances_)

    def test_get_feature_importance_booster(self, mock_lgbm_booster):
        """Test get_feature_importance with booster."""
        wrapper = LightGBMWrapper(mock_lgbm_booster)
        
        importance = wrapper.get_feature_importance("gain")
        
        mock_lgbm_booster.feature_importance.assert_called_once_with(
            importance_type="gain"
        )
        assert len(importance) == 4

    def test_get_feature_importance_invalid_type(self, mock_lgbm_sklearn_model):
        """Test get_feature_importance with invalid type."""
        wrapper = LightGBMWrapper(mock_lgbm_sklearn_model)
        
        with pytest.raises(ValueError, match="Invalid importance_type"):
            wrapper.get_feature_importance("invalid")

    def test_get_feature_names_sklearn(self, mock_lgbm_sklearn_model):
        """Test get_feature_names with sklearn model."""
        wrapper = LightGBMWrapper(mock_lgbm_sklearn_model)
        
        names = wrapper.get_feature_names()
        
        assert names == ["f1", "f2", "f3", "f4", "f5"]

    def test_get_feature_names_booster(self, mock_lgbm_booster):
        """Test get_feature_names with booster."""
        wrapper = LightGBMWrapper(mock_lgbm_booster)
        
        names = wrapper.get_feature_names()
        
        mock_lgbm_booster.feature_name.assert_called_once()
        assert names == ["feat1", "feat2", "feat3", "feat4"]

    def test_get_model_type(self, mock_lgbm_sklearn_model):
        """Test get_model_type."""
        wrapper = LightGBMWrapper(mock_lgbm_sklearn_model)
        
        assert wrapper.get_model_type() == "lightgbm"

    def test_get_n_classes_classifier(self, mock_lgbm_sklearn_model):
        """Test get_n_classes for classifier."""
        wrapper = LightGBMWrapper(mock_lgbm_sklearn_model)
        
        assert wrapper.get_n_classes() == 2

    def test_get_n_classes_regressor(self, mock_lgbm_booster):
        """Test get_n_classes for regressor."""
        wrapper = LightGBMWrapper(mock_lgbm_booster)
        
        assert wrapper.get_n_classes() is None

    def test_is_classifier(self, mock_lgbm_sklearn_model):
        """Test is_classifier method."""
        wrapper = LightGBMWrapper(mock_lgbm_sklearn_model)
        
        assert wrapper.is_classifier() is True

    def test_get_objective_sklearn(self, mock_lgbm_sklearn_model):
        """Test get_objective with sklearn model."""
        wrapper = LightGBMWrapper(mock_lgbm_sklearn_model)
        
        assert wrapper.get_objective() == "binary"

    def test_get_objective_booster(self, mock_lgbm_booster):
        """Test get_objective with booster."""
        wrapper = LightGBMWrapper(mock_lgbm_booster)
        
        assert wrapper.get_objective() == "regression"

    def test_get_n_iterations(self, mock_lgbm_booster):
        """Test get_n_iterations."""
        wrapper = LightGBMWrapper(mock_lgbm_booster)
        
        iterations = wrapper.get_n_iterations()
        
        assert iterations == 100


class TestLightGBMAnalyzer:
    """Test cases for LightGBMAnalyzer class."""

    @pytest.fixture
    def mock_lgbm_model(self):
        """Create mock LightGBM model for analyzer."""
        model = Mock()
        model.__class__.__name__ = "LGBMClassifier"
        model.__class__.__module__ = "lightgbm.sklearn"
        model.n_features_ = 3
        model.n_classes_ = 2
        model.objective_ = "binary"
        model.feature_name_ = ["feature_0", "feature_1", "feature_2"]
        model.feature_importances_ = np.array([0.5, 0.3, 0.2])
        model.n_estimators = 100
        
        # Mock methods
        model.predict = Mock(return_value=np.array([0, 1, 0, 1, 0]))
        model.predict_proba = Mock(return_value=np.array([
            [0.8, 0.2], [0.3, 0.7], [0.6, 0.4], [0.2, 0.8], [0.9, 0.1]
        ]))
        model.get_params = Mock(return_value={
            "boosting_type": "gbdt",
            "n_estimators": 100,
            "learning_rate": 0.1,
            "num_leaves": 31,
        })
        
        return model

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        X = pd.DataFrame({
            "feature_0": [1, 2, 3, 4, 5],
            "feature_1": [5, 4, 3, 2, 1],
            "feature_2": [2, 3, 4, 5, 6],
        })
        y = pd.Series([0, 1, 0, 1, 0])
        return X, y

    def test_analyzer_initialization(self, mock_lgbm_model):
        """Test analyzer initialization."""
        analyzer = LightGBMAnalyzer(mock_lgbm_model)
        
        assert analyzer.model is mock_lgbm_model
        assert isinstance(analyzer.wrapper, LightGBMWrapper)

    def test_get_model_type(self, mock_lgbm_model):
        """Test get_model_type method."""
        analyzer = LightGBMAnalyzer(mock_lgbm_model)
        
        assert analyzer.get_model_type() == "lightgbm"

    @patch('model_improvement_toolkit.core.metrics.calculate_metrics')
    def test_analyze_basic(self, mock_calculate_metrics, mock_lgbm_model, sample_data):
        """Test basic analyze functionality."""
        X, y = sample_data
        
        # Mock metrics calculation
        mock_calculate_metrics.return_value = {
            "accuracy": 0.8,
            "precision": 0.75,
            "recall": 0.8,
            "f1": 0.77,
        }
        
        analyzer = LightGBMAnalyzer(mock_lgbm_model)
        results = analyzer.analyze(X, y)
        
        # Check results structure
        assert "model_info" in results
        assert "feature_importance" in results
        assert "performance_metrics" in results
        assert "overfitting_analysis" in results
        assert "suggestions" in results

    def test_extract_model_info(self, mock_lgbm_model, sample_data):
        """Test _extract_model_info method."""
        X, y = sample_data
        analyzer = LightGBMAnalyzer(mock_lgbm_model)
        
        model_info = analyzer._extract_model_info()
        
        assert model_info["model_type"] == "lightgbm"
        assert model_info["n_features"] == 3
        assert model_info["n_iterations"] == 100
        assert model_info["objective"] == "binary"
        assert model_info["is_classifier"] is True
        assert model_info["n_classes"] == 2
        assert "parameters" in model_info

    def test_analyze_feature_importance(self, mock_lgbm_model, sample_data):
        """Test _analyze_feature_importance method."""
        X, y = sample_data
        analyzer = LightGBMAnalyzer(mock_lgbm_model)
        
        importance_analysis = analyzer._analyze_feature_importance()
        
        assert "importance_scores" in importance_analysis
        assert "top_features" in importance_analysis
        assert "low_importance_features" in importance_analysis
        assert "cumulative_importance_90" in importance_analysis
        
        # Check that features are ranked by importance
        scores = importance_analysis["importance_scores"]
        assert len(scores) == 3
        assert scores[0]["feature"] == "feature_0"  # Highest importance

    @patch('model_improvement_toolkit.core.metrics.calculate_metrics')
    def test_calculate_metrics(self, mock_calculate_metrics, mock_lgbm_model, sample_data):
        """Test _calculate_metrics method."""
        X, y = sample_data
        X_val, y_val = X.copy(), y.copy()
        
        # Mock metrics
        mock_calculate_metrics.return_value = {
            "accuracy": 0.8,
            "precision": 0.75,
            "auc": 0.85,
        }
        
        analyzer = LightGBMAnalyzer(mock_lgbm_model)
        metrics = analyzer._calculate_metrics(X, y, X_val, y_val)
        
        assert "train" in metrics
        assert "validation" in metrics
        assert mock_calculate_metrics.call_count >= 2

    def test_diagnose_overfitting_no_validation(self, mock_lgbm_model, sample_data):
        """Test overfitting diagnosis without validation data."""
        X, y = sample_data
        analyzer = LightGBMAnalyzer(mock_lgbm_model)
        
        diagnosis = analyzer._diagnose_overfitting(X, y)
        
        assert diagnosis["has_validation"] is False
        assert diagnosis["overfitting_detected"] is False
        assert "message" in diagnosis

    def test_diagnose_overfitting_with_validation(self, mock_lgbm_model, sample_data):
        """Test overfitting diagnosis with validation data."""
        X, y = sample_data
        X_val, y_val = X.copy(), y.copy()
        
        analyzer = LightGBMAnalyzer(mock_lgbm_model)
        
        # Mock performance metrics
        analyzer._results = {
            "performance_metrics": {
                "train": {"accuracy": 0.95},
                "validation": {"accuracy": 0.70},  # Large gap
            }
        }
        
        diagnosis = analyzer._diagnose_overfitting(X, y, X_val, y_val)
        
        assert diagnosis["has_validation"] is True
        assert diagnosis["overfitting_detected"] is True
        assert diagnosis["severity"] == "high"
        assert diagnosis["metrics"]["score_gap"] == 0.25

    def test_generate_suggestions_overfitting(self, mock_lgbm_model, sample_data):
        """Test suggestion generation for overfitting."""
        X, y = sample_data
        analyzer = LightGBMAnalyzer(mock_lgbm_model)
        
        # Mock analysis results
        analyzer._results = {
            "overfitting_analysis": {
                "overfitting_detected": True,
                "severity": "high",
            },
            "feature_importance": {
                "low_importance_features": ["feat1", "feat2", "feat3", "feat4", "feat5", "feat6"],
            },
            "model_info": {
                "parameters": {"learning_rate": 0.2},
            },
        }
        
        suggestions = analyzer._generate_suggestions()
        
        assert len(suggestions) >= 2  # Overfitting + feature selection + hyperparameter
        
        # Check overfitting suggestion
        overfitting_suggestion = next(
            (s for s in suggestions if s["type"] == "overfitting"), None
        )
        assert overfitting_suggestion is not None
        assert overfitting_suggestion["priority"] == "high"

    def test_generate_suggestions_feature_selection(self, mock_lgbm_model, sample_data):
        """Test suggestion generation for feature selection."""
        X, y = sample_data
        analyzer = LightGBMAnalyzer(mock_lgbm_model)
        
        # Mock many low importance features
        low_importance_features = [f"feat_{i}" for i in range(10)]
        analyzer._results = {
            "overfitting_analysis": {"overfitting_detected": False},
            "feature_importance": {
                "low_importance_features": low_importance_features,
            },
            "model_info": {
                "parameters": {"learning_rate": 0.1},
            },
        }
        
        suggestions = analyzer._generate_suggestions()
        
        # Check feature selection suggestion
        feature_suggestion = next(
            (s for s in suggestions if s["type"] == "feature_selection"), None
        )
        assert feature_suggestion is not None
        assert feature_suggestion["priority"] == "medium"
        assert "10 features have very low importance" in feature_suggestion["message"]

    def test_invalid_model_initialization(self):
        """Test analyzer initialization with invalid model."""
        invalid_model = Mock()
        invalid_model.__class__.__name__ = "RandomForestClassifier"
        invalid_model.__class__.__module__ = "sklearn.ensemble"
        
        with pytest.raises(UnsupportedModelError):
            LightGBMAnalyzer(invalid_model)