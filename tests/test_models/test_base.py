"""Tests for BaseModelWrapper class."""

from typing import Any, Dict, List
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from model_improvement_toolkit.models.base import BaseModelWrapper
from model_improvement_toolkit.utils.exceptions import UnsupportedModelError


class ConcreteModelWrapper(BaseModelWrapper):
    """Concrete implementation for testing."""

    def _validate_model(self) -> None:
        """Validate model type."""
        if not hasattr(self.model, "model_type"):
            raise UnsupportedModelError(
                "unknown",
                ["test_model"]
            )

    def predict(self, X, **kwargs):
        """Mock predict method."""
        return np.array([1, 0, 1])

    def predict_proba(self, X, **kwargs):
        """Mock predict_proba method."""
        return np.array([[0.3, 0.7], [0.8, 0.2], [0.4, 0.6]])

    def get_params(self):
        """Mock get_params method."""
        return {"param1": 1, "param2": "value"}

    def get_feature_importance(self, importance_type="gain"):
        """Mock get_feature_importance method."""
        if importance_type not in ["gain", "split"]:
            raise ValueError(f"Unknown importance type: {importance_type}")
        return np.array([0.5, 0.3, 0.2])

    def get_feature_names(self):
        """Mock get_feature_names method."""
        return ["feature1", "feature2", "feature3"]

    def get_model_type(self):
        """Mock get_model_type method."""
        return "test_model"

    def get_n_classes(self):
        """Mock get_n_classes method."""
        return 2


class RegressionModelWrapper(ConcreteModelWrapper):
    """Regression model wrapper for testing."""

    def get_n_classes(self):
        """Return None for regression."""
        return None

    def predict_proba(self, X, **kwargs):
        """Raise error for regression."""
        raise NotImplementedError("Regression models don't support predict_proba")


class TestBaseModelWrapper:
    """Test cases for BaseModelWrapper class."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock model."""
        model = Mock()
        model.model_type = "test_model"
        return model

    @pytest.fixture
    def wrapper(self, mock_model):
        """Create a wrapper instance."""
        return ConcreteModelWrapper(mock_model)

    def test_initialization(self, mock_model):
        """Test BaseModelWrapper initialization."""
        wrapper = ConcreteModelWrapper(mock_model)
        assert wrapper.model is mock_model

    def test_initialization_invalid_model(self):
        """Test initialization with invalid model."""
        invalid_model = Mock()
        # Remove model_type attribute
        if hasattr(invalid_model, "model_type"):
            delattr(invalid_model, "model_type")
        
        with pytest.raises(UnsupportedModelError):
            ConcreteModelWrapper(invalid_model)

    def test_predict(self, wrapper):
        """Test predict method."""
        X = pd.DataFrame({"f1": [1, 2, 3]})
        predictions = wrapper.predict(X)
        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == 3

    def test_predict_proba(self, wrapper):
        """Test predict_proba method."""
        X = pd.DataFrame({"f1": [1, 2, 3]})
        probas = wrapper.predict_proba(X)
        assert isinstance(probas, np.ndarray)
        assert probas.shape == (3, 2)

    def test_get_params(self, wrapper):
        """Test get_params method."""
        params = wrapper.get_params()
        assert isinstance(params, dict)
        assert "param1" in params
        assert params["param1"] == 1

    def test_get_feature_importance(self, wrapper):
        """Test get_feature_importance method."""
        importance = wrapper.get_feature_importance()
        assert isinstance(importance, np.ndarray)
        assert len(importance) == 3
        
        # Test with different importance type
        importance_split = wrapper.get_feature_importance("split")
        assert isinstance(importance_split, np.ndarray)

    def test_get_feature_importance_invalid_type(self, wrapper):
        """Test get_feature_importance with invalid type."""
        with pytest.raises(ValueError, match="Unknown importance type"):
            wrapper.get_feature_importance("invalid_type")

    def test_get_feature_names(self, wrapper):
        """Test get_feature_names method."""
        names = wrapper.get_feature_names()
        assert isinstance(names, list)
        assert len(names) == 3
        assert names[0] == "feature1"

    def test_get_model_type(self, wrapper):
        """Test get_model_type method."""
        model_type = wrapper.get_model_type()
        assert model_type == "test_model"

    def test_get_n_features(self, wrapper):
        """Test get_n_features method."""
        n_features = wrapper.get_n_features()
        assert n_features == 3

    def test_get_n_classes(self, wrapper):
        """Test get_n_classes method."""
        n_classes = wrapper.get_n_classes()
        assert n_classes == 2

    def test_is_classifier(self, wrapper):
        """Test is_classifier method."""
        assert wrapper.is_classifier() is True

    def test_is_classifier_regression(self, mock_model):
        """Test is_classifier for regression model."""
        wrapper = RegressionModelWrapper(mock_model)
        assert wrapper.is_classifier() is False

    def test_get_objective(self, wrapper):
        """Test get_objective method."""
        objective = wrapper.get_objective()
        assert objective is None  # Default implementation

    def test_abstract_methods(self):
        """Test that abstract methods must be implemented."""
        with pytest.raises(TypeError):
            # Cannot instantiate abstract class
            BaseModelWrapper(Mock())  # type: ignore

    def test_regression_model_predict_proba(self, mock_model):
        """Test that regression models raise error for predict_proba."""
        wrapper = RegressionModelWrapper(mock_model)
        X = pd.DataFrame({"f1": [1, 2, 3]})
        
        with pytest.raises(NotImplementedError, match="Regression models don't support predict_proba"):
            wrapper.predict_proba(X)