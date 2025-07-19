# Model Improvement Toolkit 設計仕様書

**作成日**: 2025-07-19
**関連要件**: [requirements.md](./requirements.md)

## 1. 設計概要

### 1.1 設計方針
- **モジュール性**: 各機能を独立したモジュールとして実装し、疎結合を保つ
- **拡張性**: プラグインアーキテクチャを採用し、新機能の追加を容易にする
- **テスタビリティ**: 各コンポーネントを単体でテスト可能な設計にする
- **パフォーマンス**: 大規模データセットでも効率的に動作するよう最適化

### 1.2 技術スタック
- **言語**: Python 3.8+
- **主要ライブラリ**:
  - LightGBM >= 3.0
  - XGBoost >= 1.5
  - scikit-learn >= 1.0
  - pandas >= 1.3
  - numpy >= 1.20
  - matplotlib >= 3.4
  - seaborn >= 0.11
  - shap >= 0.40
- **開発ツール**:
  - pytest (テスト)
  - black (フォーマッター)
  - mypy (型チェック)
  - sphinx (ドキュメント生成)

## 2. パッケージ構造

```
model_improvement_toolkit/
├── __init__.py
├── core/
│   ├── __init__.py
│   ├── analyzer.py         # モデル分析の基底クラス
│   ├── metrics.py          # メトリクス計算
│   └── validator.py        # データ検証
├── models/
│   ├── __init__.py
│   ├── base.py            # モデルラッパーの基底クラス
│   ├── lightgbm.py        # LightGBM専用の実装
│   └── xgboost.py         # XGBoost専用の実装
├── analysis/
│   ├── __init__.py
│   ├── feature_importance.py   # 特徴量重要度分析
│   ├── overfitting.py         # 過学習診断
│   ├── error_analysis.py      # 誤差分析
│   └── shap_analysis.py       # SHAP値分析
├── suggestions/
│   ├── __init__.py
│   ├── feature_engineering.py  # 特徴量エンジニアリング提案
│   ├── hyperparameter.py       # ハイパーパラメータ提案
│   └── ensemble.py             # アンサンブル提案
├── visualization/
│   ├── __init__.py
│   ├── plots.py               # 基本的なプロット
│   └── report_generator.py    # レポート生成
├── utils/
│   ├── __init__.py
│   ├── data_utils.py          # データ処理ユーティリティ
│   └── config.py              # 設定管理
└── tests/
    ├── __init__.py
    ├── test_core/
    ├── test_models/
    ├── test_analysis/
    └── test_suggestions/
```

## 3. 主要クラス設計

### 3.1 ModelAnalyzer (core/analyzer.py)
```python
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union
import pandas as pd

class ModelAnalyzer(ABC):
    """モデル分析の基底クラス"""
    
    def __init__(self, model: Any, feature_names: Optional[List[str]] = None):
        self.model = model
        self.feature_names = feature_names
        self._results: Dict[str, Any] = {}
    
    @abstractmethod
    def analyze(self, X: pd.DataFrame, y: pd.Series, 
                X_val: Optional[pd.DataFrame] = None, 
                y_val: Optional[pd.Series] = None) -> Dict[str, Any]:
        """モデルの包括的な分析を実行"""
        pass
    
    @abstractmethod
    def get_model_type(self) -> str:
        """モデルタイプを返す"""
        pass
    
    def generate_report(self, output_format: str = "html") -> str:
        """分析結果からレポートを生成"""
        pass
```

### 3.2 LightGBMAnalyzer (models/lightgbm.py)
```python
from ..core.analyzer import ModelAnalyzer
import lightgbm as lgb

class LightGBMAnalyzer(ModelAnalyzer):
    """LightGBM専用のアナライザー"""
    
    def __init__(self, model: lgb.Booster):
        super().__init__(model)
        self._validate_model(model)
    
    def analyze(self, X: pd.DataFrame, y: pd.Series, 
                X_val: Optional[pd.DataFrame] = None, 
                y_val: Optional[pd.Series] = None) -> Dict[str, Any]:
        """LightGBMモデルの分析を実行"""
        results = {
            "model_info": self._extract_model_info(),
            "feature_importance": self._analyze_feature_importance(),
            "performance_metrics": self._calculate_metrics(X, y, X_val, y_val),
            "overfitting_analysis": self._diagnose_overfitting(X, y, X_val, y_val),
            "suggestions": self._generate_suggestions()
        }
        self._results = results
        return results
```

### 3.3 FeatureImportanceAnalyzer (analysis/feature_importance.py)
```python
from typing import List, Tuple, Dict, Any
import pandas as pd
import numpy as np

class FeatureImportanceAnalyzer:
    """特徴量重要度の詳細分析"""
    
    def __init__(self, importance_type: str = "gain"):
        self.importance_type = importance_type
    
    def analyze(self, model: Any, feature_names: List[str]) -> Dict[str, Any]:
        """特徴量重要度を分析"""
        importance_scores = self._get_importance_scores(model)
        return {
            "importance_ranking": self._rank_features(importance_scores, feature_names),
            "cumulative_importance": self._calculate_cumulative_importance(importance_scores),
            "low_importance_features": self._identify_low_importance_features(importance_scores, feature_names),
            "importance_stability": self._check_importance_stability(model)
        }
    
    def suggest_feature_selection(self, importance_data: Dict[str, Any], 
                                threshold: float = 0.95) -> List[str]:
        """重要度に基づいた特徴量選択を提案"""
        pass
```

### 3.4 HyperparameterOptimizer (suggestions/hyperparameter.py)
```python
from typing import Dict, Any, List, Tuple
import optuna

class HyperparameterOptimizer:
    """ハイパーパラメータ最適化の提案"""
    
    def __init__(self, model_type: str, current_params: Dict[str, Any]):
        self.model_type = model_type
        self.current_params = current_params
        self.search_space = self._define_search_space()
    
    def suggest_parameters(self, performance_history: Dict[str, float]) -> Dict[str, Any]:
        """現在の性能に基づいてパラメータを提案"""
        return {
            "recommended_params": self._get_recommended_params(),
            "parameter_importance": self._analyze_parameter_importance(),
            "expected_improvement": self._estimate_improvement(),
            "optimization_strategy": self._suggest_optimization_strategy()
        }
    
    def _define_search_space(self) -> Dict[str, Any]:
        """モデルタイプに応じた探索空間を定義"""
        if self.model_type == "lightgbm":
            return {
                "num_leaves": (20, 300),
                "learning_rate": (0.01, 0.3),
                "feature_fraction": (0.5, 1.0),
                "bagging_fraction": (0.5, 1.0),
                "lambda_l1": (0, 10),
                "lambda_l2": (0, 10)
            }
        elif self.model_type == "xgboost":
            return {
                "max_depth": (3, 10),
                "learning_rate": (0.01, 0.3),
                "subsample": (0.5, 1.0),
                "colsample_bytree": (0.5, 1.0),
                "reg_alpha": (0, 10),
                "reg_lambda": (0, 10)
            }
```

### 3.5 ReportGenerator (visualization/report_generator.py)
```python
from typing import Dict, Any
import jinja2
import matplotlib.pyplot as plt
import seaborn as sns

class ReportGenerator:
    """分析レポートの生成"""
    
    def __init__(self, template_dir: str = "templates"):
        self.template_loader = jinja2.FileSystemLoader(template_dir)
        self.template_env = jinja2.Environment(loader=self.template_loader)
    
    def generate_html_report(self, analysis_results: Dict[str, Any], 
                           output_path: str) -> str:
        """HTML形式のレポートを生成"""
        template = self.template_env.get_template("report_template.html")
        
        # プロットの生成
        plots = self._generate_plots(analysis_results)
        
        # レポートのレンダリング
        html_content = template.render(
            results=analysis_results,
            plots=plots,
            timestamp=datetime.now().isoformat()
        )
        
        with open(output_path, "w") as f:
            f.write(html_content)
        
        return output_path
    
    def _generate_plots(self, results: Dict[str, Any]) -> Dict[str, str]:
        """分析結果から各種プロットを生成"""
        plots = {}
        
        # 特徴量重要度プロット
        if "feature_importance" in results:
            plots["feature_importance"] = self._plot_feature_importance(
                results["feature_importance"]
            )
        
        # 学習曲線
        if "learning_curves" in results:
            plots["learning_curves"] = self._plot_learning_curves(
                results["learning_curves"]
            )
        
        return plots
```

## 4. API設計

### 4.1 基本的な使用方法
```python
from model_improvement_toolkit import analyze_model

# モデルの分析
results = analyze_model(
    model=trained_model,
    X_train=X_train,
    y_train=y_train,
    X_val=X_val,
    y_val=y_val,
    model_type="lightgbm"  # or "xgboost"
)

# レポートの生成
report_path = results.generate_report(
    output_format="html",
    output_path="analysis_report.html"
)

# 特定の分析のみ実行
from model_improvement_toolkit.analysis import FeatureImportanceAnalyzer

analyzer = FeatureImportanceAnalyzer()
importance_results = analyzer.analyze(model, feature_names)
```

### 4.2 高度な使用方法
```python
from model_improvement_toolkit import ModelImprovementToolkit

# カスタム設定でツールキットを初期化
toolkit = ModelImprovementToolkit(
    config={
        "analysis": {
            "include_shap": True,
            "cross_validation_folds": 5
        },
        "suggestions": {
            "hyperparameter_trials": 100,
            "feature_selection_threshold": 0.95
        }
    }
)

# 包括的な分析と改善提案
improvement_plan = toolkit.create_improvement_plan(
    model=model,
    data={"train": (X_train, y_train), "val": (X_val, y_val)},
    objective="maximize_f1",
    time_budget=3600  # 1時間
)

# 提案の実行
improved_model = toolkit.apply_suggestions(
    model=model,
    suggestions=improvement_plan.top_suggestions(n=5)
)
```

## 5. エラーハンドリング

### 5.1 カスタム例外クラス
```python
class ModelImprovementError(Exception):
    """基底例外クラス"""
    pass

class UnsupportedModelError(ModelImprovementError):
    """サポートされていないモデルタイプ"""
    pass

class InvalidDataError(ModelImprovementError):
    """無効なデータ形式"""
    pass

class AnalysisError(ModelImprovementError):
    """分析中のエラー"""
    pass
```

### 5.2 エラーハンドリング方針
- 明確なエラーメッセージで問題を特定
- 可能な限り部分的な結果を返す
- ログ出力で詳細な診断情報を提供

## 6. テスト戦略

### 6.1 単体テスト
- 各モジュールの機能を独立してテスト
- モックを使用して外部依存を排除
- エッジケースと異常系を網羅

### 6.2 統合テスト
- エンドツーエンドのワークフローをテスト
- 実際のモデルとデータセットを使用
- パフォーマンステストを含む

### 6.3 テストデータ
- 合成データセットでの基本動作確認
- 公開データセット（iris、bostonなど）での検証
- 大規模データセットでのパフォーマンステスト