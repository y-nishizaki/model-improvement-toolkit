"""Model Improvement Toolkit

機械学習モデル（LightGBMとXGBoost）の改善のための示唆を提供するPythonパッケージ
"""

__version__ = "0.1.0"
__author__ = "Model Improvement Toolkit Contributors"
__email__ = "contact@example.com"

from typing import Any, Dict, Optional, Union
import pandas as pd

from .core.validator import ModelValidator
from .models.lightgbm import LightGBMAnalyzer
from .utils.exceptions import UnsupportedModelError


class AnalysisResults:
    """Container for analysis results with convenience methods."""
    
    def __init__(self, results: Dict[str, Any], analyzer: Any):
        """
        Initialize AnalysisResults.
        
        Parameters
        ----------
        results : Dict[str, Any]
            Raw analysis results
        analyzer : Any
            Analyzer instance used for analysis
        """
        self._results = results
        self._analyzer = analyzer
    
    def get_results(self) -> Dict[str, Any]:
        """Get raw analysis results."""
        return self._results
    
    def generate_report(
        self,
        output_path: str,
        output_format: str = "html",
        **kwargs
    ) -> str:
        """
        Generate analysis report.
        
        Parameters
        ----------
        output_path : str
            Output file path
        output_format : str, default="html"
            Report format ("html", "text")
        **kwargs
            Additional arguments for report generation
        
        Returns
        -------
        str
            Path to generated report
        """
        from .visualization.report_generator import ReportGenerator
        
        generator = ReportGenerator()
        
        if output_format == "html":
            return generator.generate_html_report(
                self._results,
                output_path,
                **kwargs
            )
        elif output_format == "text":
            summary = generator.generate_summary_text(self._results)
            with open(output_path, "w") as f:
                f.write(summary)
            return output_path
        else:
            raise ValueError(f"Unsupported format: {output_format}")
    
    def get_top_features(self, n: int = 10) -> list:
        """Get top N important features."""
        feature_importance = self._results.get("feature_importance", {})
        return feature_importance.get("top_features", [])[:n]
    
    def get_suggestions(self, priority: Optional[str] = None) -> list:
        """Get improvement suggestions, optionally filtered by priority."""
        suggestions = self._results.get("suggestions", [])
        if priority:
            return [s for s in suggestions if s.get("priority") == priority]
        return suggestions
    
    def __repr__(self) -> str:
        """String representation."""
        model_info = self._results.get("model_info", {})
        return (
            f"AnalysisResults(model_type={model_info.get('model_type')}, "
            f"n_features={model_info.get('n_features')}, "
            f"suggestions={len(self.get_suggestions())})"
        )


def analyze_model(
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: Optional[pd.DataFrame] = None,
    y_val: Optional[pd.Series] = None,
    model_type: Optional[str] = None,
    feature_names: Optional[list] = None,
) -> AnalysisResults:
    """
    モデルの包括的な分析を実行
    
    Parameters
    ----------
    model : Any
        学習済みのLightGBMまたはXGBoostモデル
    X_train : pd.DataFrame
        訓練データの特徴量
    y_train : pd.Series
        訓練データのターゲット
    X_val : pd.DataFrame, optional
        検証データの特徴量
    y_val : pd.Series, optional
        検証データのターゲット
    model_type : str, optional
        モデルタイプ ("lightgbm" or "xgboost")。指定しない場合は自動検出
    feature_names : list, optional
        特徴量名のリスト
    
    Returns
    -------
    AnalysisResults
        分析結果を含むオブジェクト
    
    Examples
    --------
    >>> import lightgbm as lgb
    >>> from model_improvement_toolkit import analyze_model
    >>> 
    >>> # Train a model
    >>> model = lgb.LGBMClassifier()
    >>> model.fit(X_train, y_train)
    >>> 
    >>> # Analyze the model
    >>> results = analyze_model(model, X_train, y_train, X_val, y_val)
    >>> 
    >>> # Generate HTML report
    >>> results.generate_report("analysis_report.html")
    >>> 
    >>> # Get top features
    >>> top_features = results.get_top_features(10)
    >>> 
    >>> # Get high priority suggestions
    >>> suggestions = results.get_suggestions(priority="high")
    """
    # Validate model type
    validator = ModelValidator()
    
    if model_type:
        detected_framework, detected_type = validator.validate(model, model_type)
    else:
        detected_framework, detected_type = validator.validate(model)
    
    # Select appropriate analyzer
    if detected_framework == "lightgbm":
        analyzer = LightGBMAnalyzer(model, feature_names=feature_names)
    elif detected_framework == "xgboost":
        # TODO: Implement XGBoostAnalyzer
        raise NotImplementedError("XGBoost support will be implemented in future version")
    else:
        raise UnsupportedModelError(
            detected_framework,
            ["lightgbm", "xgboost"]
        )
    
    # Run analysis
    results = analyzer.analyze(X_train, y_train, X_val, y_val)
    
    # Return wrapped results
    return AnalysisResults(results, analyzer)


__all__ = ["analyze_model", "AnalysisResults", "__version__"]