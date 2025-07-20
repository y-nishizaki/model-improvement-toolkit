"""Custom exceptions for Model Improvement Toolkit."""


class ModelImprovementError(Exception):
    """Base exception class for all Model Improvement Toolkit errors."""

    pass


class UnsupportedModelError(ModelImprovementError):
    """Raised when an unsupported model type is provided."""

    def __init__(self, model_type: str, supported_types: list[str]) -> None:
        """
        Initialize UnsupportedModelError.

        Parameters
        ----------
        model_type : str
            The provided model type
        supported_types : list[str]
            List of supported model types
        """
        self.model_type = model_type
        self.supported_types = supported_types
        message = (
            f"Model type '{model_type}' is not supported. "
            f"Supported types are: {', '.join(supported_types)}"
        )
        super().__init__(message)


class InvalidDataError(ModelImprovementError):
    """Raised when invalid data is provided."""

    pass


class DataValidationError(InvalidDataError):
    """Raised when data validation fails."""

    def __init__(self, validation_errors: dict[str, str]) -> None:
        """
        Initialize DataValidationError.

        Parameters
        ----------
        validation_errors : dict[str, str]
            Dictionary of validation errors with field names as keys
        """
        self.validation_errors = validation_errors
        message = "Data validation failed:\n"
        for field, error in validation_errors.items():
            message += f"  - {field}: {error}\n"
        super().__init__(message.rstrip())


class AnalysisError(ModelImprovementError):
    """Raised when an error occurs during analysis."""

    pass


class InsufficientDataError(AnalysisError):
    """Raised when there is insufficient data for analysis."""

    def __init__(self, min_samples: int, actual_samples: int) -> None:
        """
        Initialize InsufficientDataError.

        Parameters
        ----------
        min_samples : int
            Minimum number of samples required
        actual_samples : int
            Actual number of samples provided
        """
        self.min_samples = min_samples
        self.actual_samples = actual_samples
        message = (
            f"Insufficient data for analysis. "
            f"Minimum samples required: {min_samples}, but got: {actual_samples}"
        )
        super().__init__(message)


class ModelCompatibilityError(AnalysisError):
    """Raised when model is incompatible with the analysis."""

    pass


class ConfigurationError(ModelImprovementError):
    """Raised when there is a configuration error."""

    pass


class InvalidConfigError(ConfigurationError):
    """Raised when configuration is invalid."""

    pass


class MissingConfigError(ConfigurationError):
    """Raised when required configuration is missing."""

    def __init__(self, config_key: str) -> None:
        """
        Initialize MissingConfigError.

        Parameters
        ----------
        config_key : str
            The missing configuration key
        """
        self.config_key = config_key
        message = f"Required configuration key '{config_key}' is missing"
        super().__init__(message)