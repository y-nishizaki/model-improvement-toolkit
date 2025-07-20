"""Configuration management for Model Improvement Toolkit."""

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml

from .exceptions import InvalidConfigError, MissingConfigError


class Config:
    """Configuration manager for Model Improvement Toolkit."""

    DEFAULT_CONFIG = {
        "analysis": {
            "include_shap": False,
            "cross_validation_folds": 5,
            "importance_threshold": 0.01,
            "min_samples": 10,
        },
        "suggestions": {
            "hyperparameter_trials": 100,
            "feature_selection_method": "recursive",
            "ensemble_max_models": 5,
        },
        "visualization": {
            "plot_style": "seaborn",
            "figure_size": [10, 6],
            "save_plots": True,
            "output_dir": "output",
        },
        "performance": {
            "n_jobs": -1,
            "memory_limit": "8GB",
            "chunk_size": 10000,
            "use_cache": True,
        },
    }

    def __init__(
        self,
        config_path: Optional[Union[str, Path]] = None,
        config_dict: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize Config.

        Parameters
        ----------
        config_path : Optional[Union[str, Path]], default=None
            Path to configuration file (YAML or JSON)
        config_dict : Optional[Dict[str, Any]], default=None
            Configuration dictionary (overrides file config)
        """
        self._config = self.DEFAULT_CONFIG.copy()

        # Load from file if provided
        if config_path is not None:
            file_config = self._load_from_file(config_path)
            self._merge_configs(self._config, file_config)

        # Load from environment variables
        env_config = self._load_from_env()
        self._merge_configs(self._config, env_config)

        # Apply dictionary config last (highest priority)
        if config_dict is not None:
            self._merge_configs(self._config, config_dict)

    def _load_from_file(self, config_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load configuration from file.

        Parameters
        ----------
        config_path : Union[str, Path]
            Path to configuration file

        Returns
        -------
        Dict[str, Any]
            Configuration dictionary

        Raises
        ------
        InvalidConfigError
            If file cannot be loaded or parsed
        """
        config_path = Path(config_path)

        if not config_path.exists():
            raise InvalidConfigError(f"Configuration file not found: {config_path}")

        try:
            with open(config_path, "r") as f:
                if config_path.suffix in [".yaml", ".yml"]:
                    return yaml.safe_load(f) or {}
                elif config_path.suffix == ".json":
                    return json.load(f)
                else:
                    raise InvalidConfigError(
                        f"Unsupported configuration file format: {config_path.suffix}"
                    )
        except (yaml.YAMLError, json.JSONDecodeError) as e:
            raise InvalidConfigError(f"Failed to parse configuration file: {e}")
        except Exception as e:
            raise InvalidConfigError(
                f"Failed to load configuration file: {e}"
            )

    def _load_from_env(self) -> Dict[str, Any]:
        """
        Load configuration from environment variables.

        Environment variables should be prefixed with MIT_
        and use double underscores for nested keys.
        Example: MIT_ANALYSIS__CROSS_VALIDATION_FOLDS=10

        Returns
        -------
        Dict[str, Any]
            Configuration dictionary from environment
        """
        env_config: Dict[str, Any] = {}
        prefix = "MIT_"

        for key, value in os.environ.items():
            if key.startswith(prefix):
                # Remove prefix and convert to lowercase
                config_key = key[len(prefix):].lower()

                # Split by double underscore for nested keys
                keys = config_key.split("__")

                # Build nested dictionary
                current = env_config
                for k in keys[:-1]:
                    if k not in current:
                        current[k] = {}
                    current = current[k]

                # Convert value type
                current[keys[-1]] = self._parse_env_value(value)

        return env_config

    def _parse_env_value(self, value: str) -> Any:
        """
        Parse environment variable value to appropriate type.

        Parameters
        ----------
        value : str
            Environment variable value

        Returns
        -------
        Any
            Parsed value
        """
        # Try to parse as JSON first (handles lists, dicts, bools, nulls)
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            pass

        # Try to parse as number
        try:
            if "." in value:
                return float(value)
            return int(value)
        except ValueError:
            pass

        # Return as string
        return value

    def _merge_configs(self, base: Dict[str, Any], update: Dict[str, Any]) -> None:
        """
        Recursively merge update config into base config.

        Parameters
        ----------
        base : Dict[str, Any]
            Base configuration dictionary (modified in place)
        update : Dict[str, Any]
            Update configuration dictionary
        """
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_configs(base[key], value)
            else:
                base[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key.

        Supports nested keys using dot notation.
        Example: "analysis.cross_validation_folds"

        Parameters
        ----------
        key : str
            Configuration key (supports dot notation)
        default : Any, default=None
            Default value if key not found

        Returns
        -------
        Any
            Configuration value
        """
        keys = key.split(".")
        value = self._config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def get_required(self, key: str) -> Any:
        """
        Get required configuration value by key.

        Parameters
        ----------
        key : str
            Configuration key (supports dot notation)

        Returns
        -------
        Any
            Configuration value

        Raises
        ------
        MissingConfigError
            If key is not found
        """
        value = self.get(key, sentinel := object())
        if value is sentinel:
            raise MissingConfigError(key)
        return value

    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value by key.

        Parameters
        ----------
        key : str
            Configuration key (supports dot notation)
        value : Any
            Value to set
        """
        keys = key.split(".")
        config = self._config

        # Navigate to the parent of the final key
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        # Set the final key
        config[keys[-1]] = value

    def to_dict(self) -> Dict[str, Any]:
        """
        Get full configuration as dictionary.

        Returns
        -------
        Dict[str, Any]
            Full configuration dictionary
        """
        import copy
        return copy.deepcopy(self._config)

    def save(self, path: Union[str, Path], format: str = "yaml") -> None:
        """
        Save configuration to file.

        Parameters
        ----------
        path : Union[str, Path]
            Output file path
        format : str, default="yaml"
            Output format ("yaml" or "json")

        Raises
        ------
        InvalidConfigError
            If format is not supported
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if format == "yaml":
            with open(path, "w") as f:
                yaml.dump(self._config, f, default_flow_style=False)
        elif format == "json":
            with open(path, "w") as f:
                json.dump(self._config, f, indent=2)
        else:
            raise InvalidConfigError(f"Unsupported format: {format}")