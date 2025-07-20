"""Pytest configuration and shared fixtures."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    np.random.seed(42)
    n_samples = 100
    n_features = 5
    
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f"feature_{i}" for i in range(n_features)]
    )
    y = pd.Series(np.random.randint(0, 2, n_samples))
    
    return X, y


@pytest.fixture
def sample_regression_data():
    """Create sample regression data for testing."""
    np.random.seed(42)
    n_samples = 100
    n_features = 5
    
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f"feature_{i}" for i in range(n_features)]
    )
    # Create target with some relationship to features
    y = pd.Series(
        X.iloc[:, 0] * 2 + X.iloc[:, 1] * -1 + np.random.randn(n_samples) * 0.1
    )
    
    return X, y


@pytest.fixture
def sample_multiclass_data():
    """Create sample multiclass data for testing."""
    np.random.seed(42)
    n_samples = 150
    n_features = 4
    n_classes = 3
    
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f"feature_{i}" for i in range(n_features)]
    )
    y = pd.Series(np.random.randint(0, n_classes, n_samples))
    
    return X, y


@pytest.fixture
def large_sample_data():
    """Create larger sample data for performance testing."""
    np.random.seed(42)
    n_samples = 10000
    n_features = 20
    
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f"feature_{i}" for i in range(n_features)]
    )
    y = pd.Series(np.random.randint(0, 2, n_samples))
    
    return X, y


@pytest.fixture
def mock_lightgbm_model():
    """Create a mock LightGBM model."""
    from unittest.mock import Mock
    
    model = Mock()
    model.__class__.__name__ = "LGBMClassifier"
    model.__class__.__module__ = "lightgbm.sklearn"
    model.n_features_ = 5
    model.n_classes_ = 2
    model.objective_ = "binary"
    
    # Mock methods
    model.predict = Mock(return_value=np.array([0, 1, 0, 1, 0]))
    model.predict_proba = Mock(return_value=np.array([
        [0.7, 0.3],
        [0.2, 0.8],
        [0.6, 0.4],
        [0.3, 0.7],
        [0.8, 0.2]
    ]))
    model.feature_importances_ = np.array([0.3, 0.25, 0.2, 0.15, 0.1])
    model.get_params = Mock(return_value={
        "boosting_type": "gbdt",
        "n_estimators": 100,
        "learning_rate": 0.1,
        "num_leaves": 31,
    })
    
    return model


@pytest.fixture
def mock_xgboost_model():
    """Create a mock XGBoost model."""
    from unittest.mock import Mock
    
    model = Mock()
    model.__class__.__name__ = "XGBRegressor"
    model.__class__.__module__ = "xgboost.sklearn"
    model.n_features_in_ = 5
    
    # Mock methods
    model.predict = Mock(return_value=np.array([1.2, 2.3, 3.4, 4.5, 5.6]))
    model.feature_importances_ = np.array([0.25, 0.3, 0.15, 0.2, 0.1])
    model.get_params = Mock(return_value={
        "booster": "gbtree",
        "n_estimators": 100,
        "learning_rate": 0.3,
        "max_depth": 6,
    })
    model.get_booster = Mock()
    
    return model


@pytest.fixture
def temp_config_file(tmp_path):
    """Create a temporary config file."""
    import yaml
    
    config_data = {
        "analysis": {
            "cross_validation_folds": 3,
            "include_shap": True,
        },
        "visualization": {
            "figure_size": [8, 6],
        }
    }
    
    config_file = tmp_path / "test_config.yaml"
    with open(config_file, "w") as f:
        yaml.dump(config_data, f)
    
    return config_file


@pytest.fixture(autouse=True)
def clean_environment():
    """Clean environment variables before and after tests."""
    import os
    
    # Store original environment
    original_env = os.environ.copy()
    
    # Remove any MIT_ prefixed variables
    for key in list(os.environ.keys()):
        if key.startswith("MIT_"):
            del os.environ[key]
    
    yield
    
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers."""
    for item in items:
        # Add unit test marker to all tests by default
        if "integration" not in item.keywords and "slow" not in item.keywords:
            item.add_marker(pytest.mark.unit)