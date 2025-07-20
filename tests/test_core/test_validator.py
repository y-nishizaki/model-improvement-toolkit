"""Tests for validator classes."""

from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from model_improvement_toolkit.core.validator import DataValidator, ModelValidator
from model_improvement_toolkit.utils.exceptions import (
    DataValidationError,
    ModelCompatibilityError,
)


class TestDataValidator:
    """Test cases for DataValidator class."""

    @pytest.fixture
    def validator(self):
        """Create a default validator."""
        return DataValidator()

    @pytest.fixture
    def valid_data(self):
        """Create valid sample data."""
        X = pd.DataFrame({
            "feature1": [1.0, 2.0, 3.0, 4.0, 5.0] * 3,
            "feature2": [5.0, 4.0, 3.0, 2.0, 1.0] * 3,
            "feature3": [2.0, 3.0, 4.0, 5.0, 6.0] * 3,
        })
        y = pd.Series([0, 1, 0, 1, 0] * 3)
        return X, y

    def test_valid_data(self, validator, valid_data):
        """Test validation with valid data."""
        X, y = valid_data
        # Should not raise any exception
        validator.validate(X, y)

    def test_numpy_array_input(self, validator):
        """Test validation with numpy array input."""
        X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]] * 5)
        y = np.array([0, 1, 0] * 5)
        # Should not raise any exception
        validator.validate(X, y)

    def test_numpy_array_with_feature_names(self, validator):
        """Test validation with numpy array and feature names."""
        X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]] * 5)
        feature_names = ["f1", "f2", "f3"]
        validator.validate(X, feature_names=feature_names)

    def test_too_few_samples(self, validator):
        """Test validation with too few samples."""
        X = pd.DataFrame({"f1": [1, 2], "f2": [3, 4]})
        with pytest.raises(DataValidationError) as exc_info:
            validator.validate(X)
        assert "Too few samples" in str(exc_info.value)

    def test_empty_dataframe(self, validator):
        """Test validation with empty DataFrame."""
        X = pd.DataFrame()
        with pytest.raises(DataValidationError) as exc_info:
            validator.validate(X)
        assert "Input data is empty" in str(exc_info.value)

    def test_missing_values_not_allowed(self, validator):
        """Test validation with missing values when not allowed."""
        X = pd.DataFrame({
            "f1": [1, 2, np.nan, 4, 5] * 3,
            "f2": [1, 2, 3, 4, 5] * 3,
        })
        with pytest.raises(DataValidationError) as exc_info:
            validator.validate(X)
        assert "missing values" in str(exc_info.value).lower()
        assert "f1" in str(exc_info.value)

    def test_missing_values_allowed(self):
        """Test validation with missing values when allowed."""
        validator = DataValidator(allow_missing=True)
        X = pd.DataFrame({
            "f1": [1, 2, np.nan, 4, 5] * 3,
            "f2": [1, 2, 3, 4, 5] * 3,
        })
        # Should not raise exception
        validator.validate(X)

    def test_too_many_missing_values(self):
        """Test validation with too many missing values."""
        validator = DataValidator(allow_missing=True, max_missing_ratio=0.1)
        X = pd.DataFrame({
            "f1": [np.nan] * 8 + [1, 2] * 3,  # >10% missing
            "f2": [1, 2, 3, 4, 5] * 4 + [np.nan, np.nan],  # <10% missing
        })
        with pytest.raises(DataValidationError) as exc_info:
            validator.validate(X)
        assert "too many missing values" in str(exc_info.value).lower()
        assert "f1" in str(exc_info.value)

    def test_infinite_values_not_allowed(self, validator):
        """Test validation with infinite values when not allowed."""
        X = pd.DataFrame({
            "f1": [1, 2, np.inf, 4, 5] * 3,
            "f2": [1, 2, 3, -np.inf, 5] * 3,
        })
        with pytest.raises(DataValidationError) as exc_info:
            validator.validate(X)
        assert "infinite values" in str(exc_info.value).lower()

    def test_infinite_values_allowed(self):
        """Test validation with infinite values when allowed."""
        validator = DataValidator(allow_infinite=True)
        X = pd.DataFrame({
            "f1": [1, 2, np.inf, 4, 5] * 3,
            "f2": [1, 2, 3, -np.inf, 5] * 3,
        })
        # Should not raise exception
        validator.validate(X)

    def test_non_numeric_features(self, validator):
        """Test validation with non-numeric features."""
        X = pd.DataFrame({
            "f1": [1, 2, 3, 4, 5] * 3,
            "f2": ["a", "b", "c", "d", "e"] * 3,  # String column
        })
        with pytest.raises(DataValidationError) as exc_info:
            validator.validate(X)
        assert "Non-numeric features" in str(exc_info.value)
        assert "f2" in str(exc_info.value)

    def test_target_with_missing_values(self, validator, valid_data):
        """Test validation with missing values in target."""
        X, _ = valid_data
        y = pd.Series([1, 0, np.nan, 1, 0] * 3)
        with pytest.raises(DataValidationError) as exc_info:
            validator.validate(X, y)
        assert "Target contains missing values" in str(exc_info.value)

    def test_target_with_infinite_values(self, validator, valid_data):
        """Test validation with infinite values in target."""
        X, _ = valid_data
        y = pd.Series([1, 0, np.inf, 1, 0] * 3)
        with pytest.raises(DataValidationError) as exc_info:
            validator.validate(X, y)
        assert "Target contains infinite values" in str(exc_info.value)

    def test_target_with_no_variance(self, validator, valid_data):
        """Test validation with constant target."""
        X, _ = valid_data
        y = pd.Series([1] * len(X))  # All same value
        with pytest.raises(DataValidationError) as exc_info:
            validator.validate(X, y)
        assert "only 1 unique value" in str(exc_info.value)

    def test_multiple_validation_errors(self, validator):
        """Test validation with multiple errors."""
        X = pd.DataFrame({
            "f1": [np.nan, np.inf, 3],  # Missing and infinite
            "f2": ["a", "b", "c"],  # Non-numeric
        })
        y = pd.Series([np.nan, 1, 1])  # Missing in target
        
        with pytest.raises(DataValidationError) as exc_info:
            validator.validate(X, y)
        
        error_str = str(exc_info.value)
        assert "Too few samples" in error_str
        assert "missing values" in error_str.lower()
        assert "infinite values" in error_str.lower()
        assert "Non-numeric features" in error_str
        assert "Target contains missing values" in error_str


class TestModelValidator:
    """Test cases for ModelValidator class."""

    @pytest.fixture
    def validator(self):
        """Create a ModelValidator instance."""
        return ModelValidator()

    def test_lightgbm_classifier_detection(self, validator):
        """Test detection of LightGBM classifier."""
        model = Mock()
        model.__class__.__name__ = "LGBMClassifier"
        model.__class__.__module__ = "lightgbm.sklearn"
        
        framework, model_type = validator.validate(model)
        assert framework == "lightgbm"
        assert model_type == "classifier"

    def test_xgboost_regressor_detection(self, validator):
        """Test detection of XGBoost regressor."""
        model = Mock()
        model.__class__.__name__ = "XGBRegressor"
        model.__class__.__module__ = "xgboost.sklearn"
        
        framework, model_type = validator.validate(model)
        assert framework == "xgboost"
        assert model_type == "regressor"

    def test_lightgbm_booster_detection(self, validator):
        """Test detection of LightGBM Booster."""
        model = Mock()
        model.__class__.__name__ = "Booster"
        model.__class__.__module__ = "lightgbm.basic"
        model.boosting_type = "gbdt"  # LightGBM specific attribute
        
        framework, model_type = validator.validate(model)
        assert framework == "lightgbm"
        assert model_type == "booster"

    def test_xgboost_booster_detection(self, validator):
        """Test detection of XGBoost Booster."""
        model = Mock()
        model.__class__.__name__ = "Booster"
        model.__class__.__module__ = "xgboost.core"
        model.get_booster = Mock()  # XGBoost specific method
        
        framework, model_type = validator.validate(model)
        assert framework == "xgboost"
        assert model_type == "booster"

    def test_expected_type_validation(self, validator):
        """Test validation with expected type."""
        model = Mock()
        model.__class__.__name__ = "LGBMClassifier"
        model.__class__.__module__ = "lightgbm.sklearn"
        
        # Correct expected type
        framework, model_type = validator.validate(model, expected_type="lightgbm")
        assert framework == "lightgbm"
        
        # Wrong expected type
        with pytest.raises(ModelCompatibilityError) as exc_info:
            validator.validate(model, expected_type="xgboost")
        assert "Expected xgboost model but got lightgbm" in str(exc_info.value)

    def test_unknown_model_type(self, validator):
        """Test with unknown model type."""
        model = Mock()
        model.__class__.__name__ = "RandomForestClassifier"
        model.__class__.__module__ = "sklearn.ensemble"
        
        with pytest.raises(ModelCompatibilityError) as exc_info:
            validator.validate(model)
        assert "Unknown model type" in str(exc_info.value)

    def test_booster_objective_detection_classifier(self, validator):
        """Test detection of classifier from booster objective."""
        model = Mock()
        model.__class__.__name__ = "Booster"
        model.__class__.__module__ = "lightgbm.basic"
        model.boosting_type = "gbdt"
        model.params = {"objective": "binary"}
        
        _, model_type = validator.validate(model)
        assert model_type == "classifier"

    def test_booster_objective_detection_regressor(self, validator):
        """Test detection of regressor from booster objective."""
        model = Mock()
        model.__class__.__name__ = "Booster"
        model.__class__.__module__ = "xgboost.core"
        model.get_booster = Mock()
        model.params = {"objective": "reg:squarederror"}
        
        _, model_type = validator.validate(model)
        assert model_type == "regressor"

    def test_check_compatibility_feature_mismatch(self, validator):
        """Test compatibility check with feature count mismatch."""
        model = Mock()
        model.n_features_ = 5
        X = pd.DataFrame(np.random.randn(10, 3))  # 3 features instead of 5
        
        with pytest.raises(ModelCompatibilityError) as exc_info:
            validator.check_compatibility(model, X)
        assert "Feature count mismatch" in str(exc_info.value)
        assert "expects 5 features but got 3" in str(exc_info.value)

    def test_check_compatibility_various_attributes(self, validator):
        """Test compatibility check with various feature count attributes."""
        X = pd.DataFrame(np.random.randn(10, 3))
        
        # Test n_features_in_
        model = Mock()
        model.n_features_in_ = 3
        validator.check_compatibility(model, X)  # Should not raise
        
        # Test num_feature method
        model = Mock()
        model.num_feature = Mock(return_value=3)
        validator.check_compatibility(model, X)  # Should not raise
        
        # Test feature_names
        model = Mock()
        model.feature_names = ["f1", "f2", "f3"]
        validator.check_compatibility(model, X)  # Should not raise

    def test_check_compatibility_no_feature_info(self, validator):
        """Test compatibility check when feature info is not available."""
        model = Mock()
        X = pd.DataFrame(np.random.randn(10, 3))
        
        # Should not raise exception when can't determine features
        validator.check_compatibility(model, X)