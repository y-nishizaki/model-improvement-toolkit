"""Data and model validation classes."""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from ..utils.exceptions import (
    DataValidationError,
    InvalidDataError,
    ModelCompatibilityError,
)


class DataValidator:
    """Validator for input data."""

    def __init__(
        self,
        allow_missing: bool = False,
        allow_infinite: bool = False,
        min_samples: int = 10,
        max_missing_ratio: float = 0.1,
    ) -> None:
        """
        Initialize DataValidator.

        Parameters
        ----------
        allow_missing : bool, default=False
            Whether to allow missing values in the data
        allow_infinite : bool, default=False
            Whether to allow infinite values in the data
        min_samples : int, default=10
            Minimum number of samples required
        max_missing_ratio : float, default=0.1
            Maximum ratio of missing values allowed per feature
        """
        self.allow_missing = allow_missing
        self.allow_infinite = allow_infinite
        self.min_samples = min_samples
        self.max_missing_ratio = max_missing_ratio

    def validate(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None,
        feature_names: Optional[List[str]] = None,
    ) -> None:
        """
        Validate input data.

        Parameters
        ----------
        X : Union[pd.DataFrame, np.ndarray]
            Feature matrix
        y : Optional[Union[pd.Series, np.ndarray]], default=None
            Target values
        feature_names : Optional[List[str]], default=None
            Feature names (used if X is numpy array)

        Raises
        ------
        DataValidationError
            If validation fails
        """
        errors = {}

        # Convert to DataFrame for easier handling
        if isinstance(X, np.ndarray):
            if feature_names is None:
                feature_names = [f"feature_{i}" for i in range(X.shape[1])]
            X_df = pd.DataFrame(X, columns=feature_names)
        else:
            X_df = X

        # Check minimum samples
        if len(X_df) < self.min_samples:
            errors["n_samples"] = (
                f"Too few samples: {len(X_df)} < {self.min_samples}"
            )

        # Check for empty DataFrame
        if X_df.empty:
            errors["data"] = "Input data is empty"

        # Check for missing values
        if not self.allow_missing:
            missing_cols = X_df.columns[X_df.isnull().any()].tolist()
            if missing_cols:
                errors["missing_values"] = (
                    f"Features with missing values: {missing_cols}"
                )
        else:
            # Check missing ratio
            missing_ratios = X_df.isnull().mean()
            high_missing = missing_ratios[missing_ratios > self.max_missing_ratio]
            if not high_missing.empty:
                errors["missing_ratio"] = (
                    f"Features with too many missing values "
                    f"(>{self.max_missing_ratio*100:.0f}%): "
                    f"{high_missing.index.tolist()}"
                )

        # Check for infinite values
        if not self.allow_infinite:
            # Check only numeric columns
            numeric_cols = X_df.select_dtypes(include=[np.number]).columns
            inf_cols = []
            for col in numeric_cols:
                if np.isinf(X_df[col].values).any():
                    inf_cols.append(col)
            if inf_cols:
                errors["infinite_values"] = (
                    f"Features with infinite values: {inf_cols}"
                )

        # Check data types
        non_numeric = X_df.select_dtypes(exclude=[np.number]).columns.tolist()
        if non_numeric:
            errors["data_types"] = (
                f"Non-numeric features found: {non_numeric}. "
                "Please encode categorical variables."
            )

        # Validate target if provided
        if y is not None:
            y_errors = self._validate_target(y)
            errors.update(y_errors)

        if errors:
            raise DataValidationError(errors)

    def _validate_target(
        self,
        y: Union[pd.Series, np.ndarray]
    ) -> Dict[str, str]:
        """
        Validate target values.

        Parameters
        ----------
        y : Union[pd.Series, np.ndarray]
            Target values

        Returns
        -------
        Dict[str, str]
            Dictionary of validation errors
        """
        errors = {}

        # Convert to numpy array for easier handling
        y_array = np.asarray(y)

        # Check for missing values
        if np.isnan(y_array).any():
            errors["target_missing"] = "Target contains missing values"

        # Check for infinite values
        if np.isinf(y_array).any():
            errors["target_infinite"] = "Target contains infinite values"

        # Check number of unique values
        n_unique = len(np.unique(y_array[~np.isnan(y_array)]))
        if n_unique < 2:
            errors["target_variance"] = (
                f"Target has only {n_unique} unique value(s)"
            )

        return errors


class ModelValidator:
    """Validator for machine learning models."""

    SUPPORTED_FRAMEWORKS = {
        "lightgbm": ["lgb", "lightgbm", "LGBMClassifier", "LGBMRegressor", "Booster"],
        "xgboost": ["xgb", "xgboost", "XGBClassifier", "XGBRegressor", "Booster"],
    }

    def __init__(self) -> None:
        """Initialize ModelValidator."""
        pass

    def validate(
        self,
        model: Any,
        expected_type: Optional[str] = None
    ) -> Tuple[str, str]:
        """
        Validate model and determine its type.

        Parameters
        ----------
        model : Any
            The model to validate
        expected_type : Optional[str], default=None
            Expected model type ("lightgbm" or "xgboost")

        Returns
        -------
        Tuple[str, str]
            (framework, model_type) where framework is "lightgbm" or "xgboost"
            and model_type is "classifier", "regressor", or "booster"

        Raises
        ------
        ModelCompatibilityError
            If model is not compatible
        """
        framework = self._detect_framework(model)
        model_type = self._detect_model_type(model, framework)

        if expected_type is not None and framework != expected_type:
            raise ModelCompatibilityError(
                f"Expected {expected_type} model but got {framework}"
            )

        return framework, model_type

    def _detect_framework(self, model: Any) -> str:
        """
        Detect the framework of the model.

        Parameters
        ----------
        model : Any
            The model to check

        Returns
        -------
        str
            Framework name ("lightgbm" or "xgboost")

        Raises
        ------
        ModelCompatibilityError
            If framework cannot be detected
        """
        model_class = model.__class__.__name__
        model_module = model.__class__.__module__

        # Check by module name
        if "lightgbm" in model_module:
            return "lightgbm"
        elif "xgboost" in model_module:
            return "xgboost"

        # Check by class name
        for framework, class_names in self.SUPPORTED_FRAMEWORKS.items():
            if model_class in class_names:
                return framework

        # Check for specific attributes
        if hasattr(model, "boosting_type"):  # LightGBM specific
            return "lightgbm"
        elif hasattr(model, "get_booster"):  # XGBoost specific
            return "xgboost"

        raise ModelCompatibilityError(
            f"Unknown model type: {model_class}. "
            f"Supported frameworks: {list(self.SUPPORTED_FRAMEWORKS.keys())}"
        )

    def _detect_model_type(self, model: Any, framework: str) -> str:
        """
        Detect whether model is classifier, regressor, or booster.

        Parameters
        ----------
        model : Any
            The model to check
        framework : str
            The framework name

        Returns
        -------
        str
            Model type ("classifier", "regressor", or "booster")
        """
        model_class = model.__class__.__name__

        if "Classifier" in model_class:
            return "classifier"
        elif "Regressor" in model_class:
            return "regressor"
        elif "Booster" in model_class:
            return "booster"

        # For booster objects, try to infer from objective
        if hasattr(model, "params"):
            params = model.params
            objective = params.get("objective", "")
            if any(obj in str(objective) for obj in ["binary", "multiclass"]):
                return "classifier"
            elif any(obj in str(objective) for obj in ["regression", "mae", "mse"]):
                return "regressor"

        # Default to booster if can't determine
        return "booster"

    def check_compatibility(
        self,
        model: Any,
        X: Union[pd.DataFrame, np.ndarray],
        task_type: Optional[str] = None
    ) -> None:
        """
        Check if model is compatible with the data.

        Parameters
        ----------
        model : Any
            The model to check
        X : Union[pd.DataFrame, np.ndarray]
            Feature matrix
        task_type : Optional[str], default=None
            Expected task type ("classification" or "regression")

        Raises
        ------
        ModelCompatibilityError
            If model is not compatible with data
        """
        # Get expected number of features
        if hasattr(model, "n_features_"):
            expected_features = model.n_features_
        elif hasattr(model, "n_features_in_"):
            expected_features = model.n_features_in_
        elif hasattr(model, "num_feature"):
            expected_features = model.num_feature()
        elif hasattr(model, "feature_names"):
            expected_features = len(model.feature_names)
        else:
            # Can't determine expected features, skip check
            return

        # Get actual number of features
        actual_features = X.shape[1]

        if expected_features != actual_features:
            raise ModelCompatibilityError(
                f"Feature count mismatch. Model expects {expected_features} "
                f"features but got {actual_features}"
            )