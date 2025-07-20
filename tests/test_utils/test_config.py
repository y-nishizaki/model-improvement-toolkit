"""Tests for configuration management."""

import json
import os
import tempfile
from pathlib import Path

import pytest
import yaml

from model_improvement_toolkit.utils.config import Config
from model_improvement_toolkit.utils.exceptions import InvalidConfigError, MissingConfigError


class TestConfig:
    """Test cases for Config class."""

    def test_default_config(self):
        """Test loading default configuration."""
        config = Config()
        
        # Check some default values
        assert config.get("analysis.cross_validation_folds") == 5
        assert config.get("suggestions.hyperparameter_trials") == 100
        assert config.get("visualization.plot_style") == "seaborn"
        assert config.get("performance.n_jobs") == -1

    def test_config_from_dict(self):
        """Test creating config from dictionary."""
        custom_config = {
            "analysis": {
                "cross_validation_folds": 10,
                "new_option": "test"
            }
        }
        config = Config(config_dict=custom_config)
        
        # Custom values should override defaults
        assert config.get("analysis.cross_validation_folds") == 10
        assert config.get("analysis.new_option") == "test"
        # Other defaults should remain
        assert config.get("analysis.include_shap") is False

    def test_load_from_yaml(self):
        """Test loading configuration from YAML file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml_content = """
            analysis:
              cross_validation_folds: 7
              include_shap: true
            visualization:
              figure_size: [12, 8]
            """
            f.write(yaml_content)
            f.flush()
            
            config = Config(config_path=f.name)
            
        os.unlink(f.name)
        
        assert config.get("analysis.cross_validation_folds") == 7
        assert config.get("analysis.include_shap") is True
        assert config.get("visualization.figure_size") == [12, 8]

    def test_load_from_json(self):
        """Test loading configuration from JSON file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json_content = {
                "analysis": {
                    "min_samples": 20
                },
                "performance": {
                    "chunk_size": 5000
                }
            }
            json.dump(json_content, f)
            f.flush()
            
            config = Config(config_path=f.name)
            
        os.unlink(f.name)
        
        assert config.get("analysis.min_samples") == 20
        assert config.get("performance.chunk_size") == 5000

    def test_load_from_env(self):
        """Test loading configuration from environment variables."""
        # Set environment variables
        os.environ["MIT_ANALYSIS__CROSS_VALIDATION_FOLDS"] = "3"
        os.environ["MIT_PERFORMANCE__N_JOBS"] = "4"
        os.environ["MIT_VISUALIZATION__SAVE_PLOTS"] = "false"
        os.environ["MIT_SUGGESTIONS__HYPERPARAMETER_TRIALS"] = "200"
        
        try:
            config = Config()
            
            assert config.get("analysis.cross_validation_folds") == 3
            assert config.get("performance.n_jobs") == 4
            assert config.get("visualization.save_plots") is False
            assert config.get("suggestions.hyperparameter_trials") == 200
        finally:
            # Clean up environment variables
            for key in list(os.environ.keys()):
                if key.startswith("MIT_"):
                    del os.environ[key]

    def test_config_priority(self):
        """Test configuration priority (file < env < dict)."""
        # Create a config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml_content = """
            analysis:
              cross_validation_folds: 7
              importance_threshold: 0.05
            """
            f.write(yaml_content)
            f.flush()
            
            # Set environment variable
            os.environ["MIT_ANALYSIS__CROSS_VALIDATION_FOLDS"] = "10"
            
            try:
                # Create config with all sources
                config = Config(
                    config_path=f.name,
                    config_dict={"analysis": {"cross_validation_folds": 15}}
                )
                
                # Dict should have highest priority
                assert config.get("analysis.cross_validation_folds") == 15
                # Env should override file
                assert config.get("analysis.importance_threshold") == 0.05
            finally:
                del os.environ["MIT_ANALYSIS__CROSS_VALIDATION_FOLDS"]
                
        os.unlink(f.name)

    def test_get_with_default(self):
        """Test get method with default value."""
        config = Config()
        
        # Existing key
        assert config.get("analysis.cross_validation_folds", 99) == 5
        
        # Non-existing key
        assert config.get("non.existing.key", "default") == "default"
        assert config.get("analysis.non_existing", 42) == 42

    def test_get_required(self):
        """Test get_required method."""
        config = Config()
        
        # Existing key
        assert config.get_required("analysis.cross_validation_folds") == 5
        
        # Non-existing key should raise
        with pytest.raises(MissingConfigError) as exc_info:
            config.get_required("non.existing.key")
        assert "non.existing.key" in str(exc_info.value)

    def test_set_value(self):
        """Test setting configuration values."""
        config = Config()
        
        # Set existing value
        config.set("analysis.cross_validation_folds", 12)
        assert config.get("analysis.cross_validation_folds") == 12
        
        # Set new value
        config.set("custom.new.value", "test")
        assert config.get("custom.new.value") == "test"
        
        # Set nested value
        config.set("analysis.custom.deeply.nested", [1, 2, 3])
        assert config.get("analysis.custom.deeply.nested") == [1, 2, 3]

    def test_to_dict(self):
        """Test converting config to dictionary."""
        config = Config(config_dict={"custom": {"value": 42}})
        config_dict = config.to_dict()
        
        # Should return a copy
        assert isinstance(config_dict, dict)
        assert config_dict["custom"]["value"] == 42
        
        # Modifying the returned dict shouldn't affect config
        config_dict["custom"]["value"] = 99
        assert config.get("custom.value") == 42

    def test_save_yaml(self):
        """Test saving configuration to YAML."""
        config = Config()
        config.set("custom.value", "test")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "config.yaml"
            config.save(output_path, format="yaml")
            
            # Load and verify
            with open(output_path) as f:
                saved_config = yaml.safe_load(f)
            
            assert saved_config["custom"]["value"] == "test"
            assert saved_config["analysis"]["cross_validation_folds"] == 5

    def test_save_json(self):
        """Test saving configuration to JSON."""
        config = Config()
        config.set("custom.value", 123)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "config.json"
            config.save(output_path, format="json")
            
            # Load and verify
            with open(output_path) as f:
                saved_config = json.load(f)
            
            assert saved_config["custom"]["value"] == 123
            assert saved_config["performance"]["n_jobs"] == -1

    def test_save_creates_directory(self):
        """Test that save creates parent directory if needed."""
        config = Config()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "nested" / "dir" / "config.yaml"
            config.save(output_path)
            
            assert output_path.exists()

    def test_invalid_config_file(self):
        """Test error handling for invalid config file."""
        # Non-existent file
        with pytest.raises(InvalidConfigError) as exc_info:
            Config(config_path="non_existent_file.yaml")
        assert "not found" in str(exc_info.value)
        
        # Invalid YAML
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("invalid: yaml: content: [")
            f.flush()
            
            with pytest.raises(InvalidConfigError) as exc_info:
                Config(config_path=f.name)
            assert "Failed to parse" in str(exc_info.value)
            
        os.unlink(f.name)
        
        # Unsupported format
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("some content")
            f.flush()
            
            with pytest.raises(InvalidConfigError) as exc_info:
                Config(config_path=f.name)
            assert "Unsupported configuration file format" in str(exc_info.value)
            
        os.unlink(f.name)

    def test_save_invalid_format(self):
        """Test error handling for invalid save format."""
        config = Config()
        
        with pytest.raises(InvalidConfigError) as exc_info:
            config.save("output.xml", format="xml")
        assert "Unsupported format: xml" in str(exc_info.value)

    def test_env_value_parsing(self):
        """Test parsing of different environment variable value types."""
        test_cases = {
            "MIT_TEST__STRING": "hello world",
            "MIT_TEST__INT": "42",
            "MIT_TEST__FLOAT": "3.14",
            "MIT_TEST__BOOL_TRUE": "true",
            "MIT_TEST__BOOL_FALSE": "false",
            "MIT_TEST__LIST": "[1, 2, 3]",
            "MIT_TEST__DICT": '{"key": "value"}',
            "MIT_TEST__NULL": "null",
        }
        
        try:
            for key, value in test_cases.items():
                os.environ[key] = value
            
            config = Config()
            
            assert config.get("test.string") == "hello world"
            assert config.get("test.int") == 42
            assert config.get("test.float") == 3.14
            assert config.get("test.bool_true") is True
            assert config.get("test.bool_false") is False
            assert config.get("test.list") == [1, 2, 3]
            assert config.get("test.dict") == {"key": "value"}
            assert config.get("test.null") is None
        finally:
            for key in test_cases:
                del os.environ[key]