"""Feature importance analysis module."""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


class FeatureImportanceAnalyzer:
    """Analyzer for feature importance."""

    def __init__(self, importance_type: str = "gain") -> None:
        """
        Initialize FeatureImportanceAnalyzer.

        Parameters
        ----------
        importance_type : str, default="gain"
            Type of importance to analyze ("gain" or "split")
        """
        self.importance_type = importance_type

    def analyze(
        self,
        model: Any,
        feature_names: List[str],
        importance_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Analyze feature importance.

        Parameters
        ----------
        model : Any
            Model with feature importance
        feature_names : List[str]
            Names of features
        importance_type : Optional[str], default=None
            Override importance type for this analysis

        Returns
        -------
        Dict[str, Any]
            Feature importance analysis results
        """
        if importance_type is None:
            importance_type = self.importance_type

        # Get importance scores
        importance_scores = self._get_importance_scores(model, importance_type)

        # Create DataFrame for analysis
        importance_df = pd.DataFrame({
            "feature": feature_names,
            "importance": importance_scores,
        })

        # Perform analysis
        results = {
            "importance_ranking": self._rank_features(importance_df),
            "cumulative_importance": self._calculate_cumulative_importance(importance_df),
            "low_importance_features": self._identify_low_importance_features(
                importance_df
            ),
            "importance_statistics": self._calculate_importance_statistics(
                importance_df
            ),
            "feature_groups": self._group_features_by_importance(importance_df),
        }

        return results

    def _get_importance_scores(
        self,
        model: Any,
        importance_type: str
    ) -> np.ndarray:
        """
        Extract importance scores from model.

        Parameters
        ----------
        model : Any
            Model object
        importance_type : str
            Type of importance

        Returns
        -------
        np.ndarray
            Importance scores
        """
        # Try different methods to get importance
        if hasattr(model, "get_feature_importance"):
            return model.get_feature_importance(importance_type)
        elif hasattr(model, "feature_importances_"):
            return model.feature_importances_
        elif hasattr(model, "feature_importance"):
            return model.feature_importance(importance_type=importance_type)
        else:
            raise AttributeError("Model doesn't have feature importance information")

    def _rank_features(self, importance_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Rank features by importance.

        Parameters
        ----------
        importance_df : pd.DataFrame
            DataFrame with feature importance

        Returns
        -------
        List[Dict[str, Any]]
            Ranked features with scores
        """
        # Sort by importance
        ranked_df = importance_df.sort_values("importance", ascending=False).copy()
        
        # Add rank
        ranked_df["rank"] = range(1, len(ranked_df) + 1)
        
        # Normalize importance (0-100 scale)
        max_importance = ranked_df["importance"].max()
        if max_importance > 0:
            ranked_df["importance_normalized"] = (
                ranked_df["importance"] / max_importance * 100
            )
        else:
            ranked_df["importance_normalized"] = 0
        
        return ranked_df.to_dict("records")

    def _calculate_cumulative_importance(
        self,
        importance_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Calculate cumulative importance.

        Parameters
        ----------
        importance_df : pd.DataFrame
            DataFrame with feature importance

        Returns
        -------
        Dict[str, Any]
            Cumulative importance analysis
        """
        # Sort by importance
        sorted_df = importance_df.sort_values("importance", ascending=False).copy()
        
        # Calculate cumulative sum
        total_importance = sorted_df["importance"].sum()
        if total_importance > 0:
            sorted_df["cumulative_importance"] = (
                sorted_df["importance"].cumsum() / total_importance
            )
        else:
            sorted_df["cumulative_importance"] = 0
        
        # Find number of features for different thresholds
        thresholds = [0.5, 0.7, 0.8, 0.9, 0.95, 0.99]
        features_needed = {}
        
        for threshold in thresholds:
            n_features = len(
                sorted_df[sorted_df["cumulative_importance"] <= threshold]
            ) + 1
            n_features = min(n_features, len(sorted_df))
            features_needed[f"features_for_{int(threshold*100)}pct"] = n_features
        
        return {
            "cumulative_importance_curve": sorted_df[
                ["feature", "cumulative_importance"]
            ].to_dict("records"),
            **features_needed,
            "total_features": len(importance_df),
        }

    def _identify_low_importance_features(
        self,
        importance_df: pd.DataFrame,
        threshold: float = 0.01,
    ) -> List[str]:
        """
        Identify features with low importance.

        Parameters
        ----------
        importance_df : pd.DataFrame
            DataFrame with feature importance
        threshold : float, default=0.01
            Threshold for low importance (relative to max)

        Returns
        -------
        List[str]
            List of low importance features
        """
        max_importance = importance_df["importance"].max()
        if max_importance == 0:
            return []
        
        # Find features below threshold
        low_importance_mask = (
            importance_df["importance"] / max_importance < threshold
        )
        
        return importance_df[low_importance_mask]["feature"].tolist()

    def _calculate_importance_statistics(
        self,
        importance_df: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Calculate statistics about feature importance distribution.

        Parameters
        ----------
        importance_df : pd.DataFrame
            DataFrame with feature importance

        Returns
        -------
        Dict[str, float]
            Importance statistics
        """
        importance_values = importance_df["importance"].values
        
        # Basic statistics
        stats = {
            "mean": float(np.mean(importance_values)),
            "std": float(np.std(importance_values)),
            "min": float(np.min(importance_values)),
            "max": float(np.max(importance_values)),
            "median": float(np.median(importance_values)),
        }
        
        # Percentiles
        for pct in [25, 75]:
            stats[f"percentile_{pct}"] = float(
                np.percentile(importance_values, pct)
            )
        
        # Coefficient of variation
        if stats["mean"] > 0:
            stats["cv"] = stats["std"] / stats["mean"]
        else:
            stats["cv"] = 0
        
        # Gini coefficient (measure of inequality)
        stats["gini"] = self._calculate_gini_coefficient(importance_values)
        
        return stats

    def _calculate_gini_coefficient(self, values: np.ndarray) -> float:
        """
        Calculate Gini coefficient for importance distribution.

        Parameters
        ----------
        values : np.ndarray
            Importance values

        Returns
        -------
        float
            Gini coefficient (0=perfect equality, 1=perfect inequality)
        """
        # Sort values
        sorted_values = np.sort(values)
        n = len(sorted_values)
        
        if n == 0 or np.sum(sorted_values) == 0:
            return 0
        
        # Calculate Gini coefficient
        index = np.arange(1, n + 1)
        return (2 * np.sum(index * sorted_values)) / (n * np.sum(sorted_values)) - (n + 1) / n

    def _group_features_by_importance(
        self,
        importance_df: pd.DataFrame,
        n_groups: int = 4,
    ) -> Dict[str, List[str]]:
        """
        Group features by importance levels.

        Parameters
        ----------
        importance_df : pd.DataFrame
            DataFrame with feature importance
        n_groups : int, default=4
            Number of groups

        Returns
        -------
        Dict[str, List[str]]
            Feature groups
        """
        # Sort by importance
        sorted_df = importance_df.sort_values("importance", ascending=False).copy()
        
        # Create groups
        group_labels = ["critical", "high", "medium", "low"][:n_groups]
        features_per_group = len(sorted_df) // n_groups
        
        groups = {}
        for i, label in enumerate(group_labels):
            start_idx = i * features_per_group
            if i == len(group_labels) - 1:
                # Last group gets remaining features
                end_idx = len(sorted_df)
            else:
                end_idx = (i + 1) * features_per_group
            
            groups[f"{label}_importance"] = sorted_df.iloc[
                start_idx:end_idx
            ]["feature"].tolist()
        
        return groups

    def suggest_feature_selection(
        self,
        importance_data: Dict[str, Any],
        threshold: float = 0.95,
        min_features: int = 5,
    ) -> Dict[str, Any]:
        """
        Suggest features to select based on importance.

        Parameters
        ----------
        importance_data : Dict[str, Any]
            Feature importance analysis results
        threshold : float, default=0.95
            Cumulative importance threshold
        min_features : int, default=5
            Minimum number of features to keep

        Returns
        -------
        Dict[str, Any]
            Feature selection suggestions
        """
        # Get cumulative importance data
        cum_importance = importance_data.get("cumulative_importance", {})
        n_features_needed = cum_importance.get(
            f"features_for_{int(threshold*100)}pct",
            len(importance_data.get("importance_ranking", []))
        )
        
        # Ensure minimum features
        n_features_needed = max(n_features_needed, min_features)
        
        # Get top features
        ranking = importance_data.get("importance_ranking", [])
        selected_features = [
            item["feature"] for item in ranking[:n_features_needed]
        ]
        
        # Calculate reduction
        total_features = cum_importance.get("total_features", len(ranking))
        reduction_pct = (
            (total_features - n_features_needed) / total_features * 100
            if total_features > 0 else 0
        )
        
        return {
            "selected_features": selected_features,
            "n_features_selected": len(selected_features),
            "n_features_removed": total_features - len(selected_features),
            "reduction_percentage": reduction_pct,
            "cumulative_importance_captured": threshold * 100,
            "removed_features": [
                item["feature"] for item in ranking[n_features_needed:]
            ],
        }