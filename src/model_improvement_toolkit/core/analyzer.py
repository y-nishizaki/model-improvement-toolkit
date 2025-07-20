"""Base analyzer class for model improvement toolkit."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import pandas as pd


class ModelAnalyzer(ABC):
    """モデル分析の基底クラス"""

    def __init__(self, model: Any, feature_names: Optional[List[str]] = None) -> None:
        """
        Initialize ModelAnalyzer.

        Parameters
        ----------
        model : Any
            学習済みモデル
        feature_names : Optional[List[str]], default=None
            特徴量名のリスト
        """
        self.model = model
        self.feature_names = feature_names
        self._results: Dict[str, Any] = {}

    @abstractmethod
    def analyze(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> Dict[str, Any]:
        """
        モデルの包括的な分析を実行.

        Parameters
        ----------
        X : pd.DataFrame
            訓練データの特徴量
        y : pd.Series
            訓練データのターゲット
        X_val : Optional[pd.DataFrame], default=None
            検証データの特徴量
        y_val : Optional[pd.Series], default=None
            検証データのターゲット

        Returns
        -------
        Dict[str, Any]
            分析結果を含む辞書
        """
        pass

    @abstractmethod
    def get_model_type(self) -> str:
        """
        モデルタイプを返す.

        Returns
        -------
        str
            モデルタイプ（"lightgbm" or "xgboost"）
        """
        pass

    def generate_report(self, output_format: str = "html") -> str:
        """
        分析結果からレポートを生成.

        Parameters
        ----------
        output_format : str, default="html"
            出力フォーマット（"html", "pdf", "notebook"）

        Returns
        -------
        str
            レポートファイルのパス

        Raises
        ------
        ValueError
            分析が実行されていない場合
        NotImplementedError
            サポートされていない出力フォーマットの場合
        """
        if not self._results:
            raise ValueError(
                "No analysis results available. Run analyze() method first."
            )

        if output_format not in ["html", "pdf", "notebook"]:
            raise NotImplementedError(f"Output format '{output_format}' is not supported.")

        # TODO: レポート生成ロジックの実装
        raise NotImplementedError("Report generation will be implemented in Phase 2")

    def get_results(self) -> Dict[str, Any]:
        """
        分析結果を取得.

        Returns
        -------
        Dict[str, Any]
            分析結果

        Raises
        ------
        ValueError
            分析が実行されていない場合
        """
        if not self._results:
            raise ValueError(
                "No analysis results available. Run analyze() method first."
            )
        return self._results

    def _validate_input(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> None:
        """
        入力データの検証.

        Parameters
        ----------
        X : pd.DataFrame
            訓練データの特徴量
        y : pd.Series
            訓練データのターゲット
        X_val : Optional[pd.DataFrame], default=None
            検証データの特徴量
        y_val : Optional[pd.Series], default=None
            検証データのターゲット

        Raises
        ------
        ValueError
            データの形状が不正な場合
        """
        if len(X) != len(y):
            raise ValueError(
                f"X and y must have the same length. Got X: {len(X)}, y: {len(y)}"
            )

        if X_val is not None and y_val is not None:
            if len(X_val) != len(y_val):
                raise ValueError(
                    f"X_val and y_val must have the same length. "
                    f"Got X_val: {len(X_val)}, y_val: {len(y_val)}"
                )

            if list(X.columns) != list(X_val.columns):
                raise ValueError(
                    "Training and validation data must have the same columns"
                )

        if self.feature_names is not None:
            if len(self.feature_names) != len(X.columns):
                raise ValueError(
                    f"Number of feature names ({len(self.feature_names)}) "
                    f"does not match number of columns ({len(X.columns)})"
                )