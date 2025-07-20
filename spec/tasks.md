# Model Improvement Toolkit 実装タスク

**関連仕様**: [requirements.md](./requirements.md) | [design.md](./design.md)
**作成日**: 2025-07-19
**最終更新**: 2025-07-20

## 全体進捗サマリー
- **Phase 1**: 基盤構築 ✅ (100%)
- **Phase 2**: コア機能実装 🟦 (20%)
- **Phase 3**: 品質保証 🟦 (30%)
- **Phase 4**: 運用準備 🔲 (0%)

## Phase 1: 基盤構築

### 1.1 プロジェクトセットアップ ✅
- [x] プロジェクト構造の作成
  - [x] src/model_improvement_toolkit/ディレクトリ構造作成
  - [x] __init__.pyファイルの配置
  - [x] 基本的なパッケージ構成の確立
- [x] 開発環境の設定
  - [x] pyproject.tomlの作成
  - [x] requirements.txtとrequirements-dev.txtの作成
  - [x] .gitignoreの設定
  - [x] pre-commitフックの設定
- [x] CI/CDパイプラインの設定
  - [x] GitHub Actionsワークフローの作成
  - [x] テスト自動実行の設定
  - [x] コードカバレッジレポートの設定

### 1.2 基底クラスとインターフェース ✅
- [x] core/analyzer.py - ModelAnalyzer基底クラスの実装
- [x] models/base.py - モデルラッパー基底クラスの実装
- [x] core/validator.py - データ検証クラスの実装
- [x] utils/config.py - 設定管理システムの実装
- [x] カスタム例外クラスの定義 (utils/exceptions.py)

### 1.3 テスト基盤 ✅
- [x] pytestの設定（pytest.ini）
- [x] テストフィクスチャの作成 (tests/conftest.py)
- [x] モックデータとモデルの準備
- [x] テストユーティリティの実装

## Phase 2: コア機能実装

### 2.1 モデル分析機能
- [x] **LightGBMサポート** ✅
  - [x] models/lightgbm.py - LightGBMAnalyzerクラスの実装
  - [x] LightGBMWrapper - sklearn/Booster API統一ラッパー
  - [x] LightGBM特有のメトリクス抽出
  - [x] テストケースの作成 (tests/test_models/test_lightgbm.py)
- [ ] **XGBoostサポート**
  - [ ] models/xgboost.py - XGBoostAnalyzerクラスの実装
  - [ ] XGBoost特有のメトリクス抽出
  - [ ] テストケースの作成
- [x] **メトリクス計算** ✅
  - [x] core/metrics.py - 基本的なメトリクス計算の実装
  - [x] 分類・回帰両方のメトリクスサポート
  - [x] 自動タスク検出機能
  - [x] テストケースの作成 (tests/test_core/test_metrics.py)

### 2.2 特徴量分析
- [x] analysis/feature_importance.py - 特徴量重要度分析の実装 ✅
  - [x] 重要度計算アルゴリズム
  - [x] 累積重要度の計算
  - [x] 低重要度特徴量の特定
  - [x] 特徴量選択提案機能
  - [x] テストケースの作成 (tests/test_analysis/test_feature_importance.py)
- [ ] analysis/error_analysis.py - 誤差分析の実装
  - [ ] 予測誤差のパターン分析
  - [ ] 誤分類サンプルの特徴分析
- [ ] 特徴量相関分析の実装
  - [ ] 高相関特徴量の検出
  - [ ] 多重共線性のチェック

### 2.3 過学習診断
- [ ] analysis/overfitting.py - 過学習診断機能の実装
  - [ ] 学習曲線とバリデーション曲線の計算
  - [ ] 過学習度の数値化
  - [ ] 早期停止の分析
- [ ] クロスバリデーション分析
  - [ ] CVスコアの分散分析
  - [ ] fold間の性能差分析

### 2.4 改善提案機能
- [ ] suggestions/feature_engineering.py - 特徴量エンジニアリング提案
  - [ ] 特徴量選択の提案
  - [ ] 特徴量変換の提案
  - [ ] 相互作用特徴量の提案
- [ ] suggestions/hyperparameter.py - ハイパーパラメータ提案
  - [ ] パラメータ探索空間の定義
  - [ ] 現在の設定からの改善提案
  - [ ] パラメータ重要度の分析
- [ ] suggestions/ensemble.py - アンサンブル提案
  - [ ] 適切なアンサンブル手法の提案
  - [ ] モデルの多様性分析

### 2.5 可視化とレポート
- [ ] visualization/plots.py - 基本的なプロット機能
  - [ ] 特徴量重要度プロット
  - [ ] 学習曲線プロット
  - [ ] 混同行列とROC曲線
- [ ] visualization/report_generator.py - レポート生成
  - [ ] HTMLテンプレートの作成
  - [ ] レポート生成ロジック
  - [ ] PDFエクスポート機能
- [ ] Jupyter Notebook用のウィジェット
  - [ ] インタラクティブなプロット
  - [ ] リアルタイム分析表示

## Phase 3: 品質保証

### 3.1 単体テスト
- [ ] 各モジュールの単体テスト作成（カバレッジ90%以上）
  - [ ] test_core/ - コアモジュールのテスト
  - [ ] test_models/ - モデルラッパーのテスト
  - [ ] test_analysis/ - 分析機能のテスト
  - [ ] test_suggestions/ - 提案機能のテスト
  - [ ] test_visualization/ - 可視化機能のテスト

### 3.2 統合テスト
- [ ] エンドツーエンドのワークフローテスト
- [ ] 異なるデータセットでの動作確認
- [ ] 大規模データでのパフォーマンステスト
- [ ] メモリ使用量のプロファイリング

### 3.3 ドキュメンテーション
- [ ] APIリファレンスの作成（Sphinx）
- [ ] ユーザーガイドの作成
- [ ] チュートリアルノートブックの作成
- [ ] サンプルコードの整備

### 3.4 コード品質
- [ ] 型ヒントの完全な実装（mypy準拠）
- [ ] コードフォーマット（black）
- [ ] Lintingエラーの解消（flake8）
- [ ] セキュリティチェック（bandit）

## Phase 4: 運用準備

### 4.1 パッケージング
- [ ] setup.pyまたはpyproject.tomlの完成
- [ ] MANIFEST.inの作成
- [ ] バージョニング戦略の決定
- [ ] ライセンスファイルの追加

### 4.2 配布準備
- [ ] PyPIへの登録準備
- [ ] ビルドとテストの自動化
- [ ] リリースワークフローの作成
- [ ] CHANGELOGの開始

### 4.3 高度な機能（オプション）
- [ ] SHAP値による説明機能の実装
- [ ] プラグインシステムの実装
- [ ] CLIツールの作成
- [ ] Webインターフェースの検討

### 4.4 コミュニティ
- [ ] コントリビューションガイドラインの作成
- [ ] イシューテンプレートの作成
- [ ] Code of Conductの追加
- [ ] 最初のリリースの準備

## 依存関係と優先順位

### 実装の依存関係
1. **Phase 1**を完了してから**Phase 2**へ
2. **Phase 2.1**（モデル分析）を優先的に実装
3. **Phase 2.2-2.4**は並行して実装可能
4. **Phase 2.5**（可視化）は他の機能完成後
5. **Phase 3**は各機能の実装と並行して進行
6. **Phase 4**は**Phase 3**がある程度進んでから

### MVP（最小限の製品）に必要なタスク ✅
1. プロジェクトセットアップ（Phase 1.1）✅
2. 基底クラス（Phase 1.2）✅
3. LightGBMサポート（Phase 2.1の一部）✅
4. 基本的な特徴量重要度分析（Phase 2.2の一部）✅
5. 簡易HTMLレポート（Phase 2.5の一部）🔲
6. 基本的なテスト（Phase 3.1の一部）✅

### 現在の実装状況（2025-07-20）
- **完了**: 基盤構築、LightGBMAnalyzer、特徴量重要度分析、メトリクス計算、テスト基盤
- **進行中**: 可視化・レポート機能
- **未着手**: XGBoostサポート、高度な分析機能、運用準備