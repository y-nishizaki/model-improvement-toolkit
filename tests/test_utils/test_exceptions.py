"""Tests for custom exceptions."""

import pytest

from model_improvement_toolkit.utils.exceptions import (
    AnalysisError,
    ConfigurationError,
    DataValidationError,
    InsufficientDataError,
    InvalidConfigError,
    InvalidDataError,
    MissingConfigError,
    ModelCompatibilityError,
    ModelImprovementError,
    UnsupportedModelError,
)


class TestModelImprovementError:
    """Test ModelImprovementError base exception."""

    def test_base_exception(self):
        """Test that base exception can be raised with message."""
        with pytest.raises(ModelImprovementError) as exc_info:
            raise ModelImprovementError("Test error message")
        assert str(exc_info.value) == "Test error message"

    def test_inheritance(self):
        """Test that ModelImprovementError inherits from Exception."""
        assert issubclass(ModelImprovementError, Exception)


class TestUnsupportedModelError:
    """Test UnsupportedModelError exception."""

    def test_exception_message(self):
        """Test exception message formatting."""
        model_type = "random_forest"
        supported_types = ["lightgbm", "xgboost"]
        
        with pytest.raises(UnsupportedModelError) as exc_info:
            raise UnsupportedModelError(model_type, supported_types)
        
        error = exc_info.value
        assert error.model_type == model_type
        assert error.supported_types == supported_types
        assert "random_forest" in str(error)
        assert "lightgbm, xgboost" in str(error)

    def test_inheritance(self):
        """Test that UnsupportedModelError inherits correctly."""
        assert issubclass(UnsupportedModelError, ModelImprovementError)


class TestDataValidationError:
    """Test DataValidationError exception."""

    def test_single_error(self):
        """Test with single validation error."""
        errors = {"column_name": "Column contains null values"}
        
        with pytest.raises(DataValidationError) as exc_info:
            raise DataValidationError(errors)
        
        error = exc_info.value
        assert error.validation_errors == errors
        assert "column_name: Column contains null values" in str(error)

    def test_multiple_errors(self):
        """Test with multiple validation errors."""
        errors = {
            "feature1": "Contains infinite values",
            "feature2": "Wrong data type",
            "target": "Too few unique values"
        }
        
        with pytest.raises(DataValidationError) as exc_info:
            raise DataValidationError(errors)
        
        error = exc_info.value
        assert error.validation_errors == errors
        error_str = str(error)
        assert "feature1: Contains infinite values" in error_str
        assert "feature2: Wrong data type" in error_str
        assert "target: Too few unique values" in error_str

    def test_inheritance(self):
        """Test inheritance chain."""
        assert issubclass(DataValidationError, InvalidDataError)
        assert issubclass(InvalidDataError, ModelImprovementError)


class TestInsufficientDataError:
    """Test InsufficientDataError exception."""

    def test_exception_message(self):
        """Test exception message formatting."""
        min_samples = 100
        actual_samples = 50
        
        with pytest.raises(InsufficientDataError) as exc_info:
            raise InsufficientDataError(min_samples, actual_samples)
        
        error = exc_info.value
        assert error.min_samples == min_samples
        assert error.actual_samples == actual_samples
        assert "Minimum samples required: 100" in str(error)
        assert "but got: 50" in str(error)

    def test_inheritance(self):
        """Test inheritance chain."""
        assert issubclass(InsufficientDataError, AnalysisError)
        assert issubclass(AnalysisError, ModelImprovementError)


class TestMissingConfigError:
    """Test MissingConfigError exception."""

    def test_exception_message(self):
        """Test exception message formatting."""
        config_key = "model.hyperparameters.learning_rate"
        
        with pytest.raises(MissingConfigError) as exc_info:
            raise MissingConfigError(config_key)
        
        error = exc_info.value
        assert error.config_key == config_key
        assert f"Required configuration key '{config_key}' is missing" in str(error)

    def test_inheritance(self):
        """Test inheritance chain."""
        assert issubclass(MissingConfigError, ConfigurationError)
        assert issubclass(ConfigurationError, ModelImprovementError)


class TestOtherExceptions:
    """Test other exception classes."""

    def test_invalid_data_error(self):
        """Test InvalidDataError."""
        with pytest.raises(InvalidDataError):
            raise InvalidDataError("Invalid data")
        assert issubclass(InvalidDataError, ModelImprovementError)

    def test_model_compatibility_error(self):
        """Test ModelCompatibilityError."""
        with pytest.raises(ModelCompatibilityError):
            raise ModelCompatibilityError("Model not compatible")
        assert issubclass(ModelCompatibilityError, AnalysisError)

    def test_invalid_config_error(self):
        """Test InvalidConfigError."""
        with pytest.raises(InvalidConfigError):
            raise InvalidConfigError("Invalid configuration")
        assert issubclass(InvalidConfigError, ConfigurationError)