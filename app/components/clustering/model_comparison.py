"""
Model comparison component for clustering results.

This module provides tools for comparing multiple clustering models
and their performance metrics.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Optional, Tuple
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from .utils import ClusteringUtils


class ModelComparison:
    """Component for comparing clustering models."""
    
    def __init__(self):
        self.utils = ClusteringUtils()
        self.comparison_metrics = [
            "silhouette_score",
            "calinski_harabasz_score", 
            "davies_bouldin_score",
            "regime_stability",
            "transition_consistency"
        ]
    
    def compare_models(self, results_list: List[Dict[str, Any]], 
                      features: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Compare multiple clustering models.
        
        Args:
            results_list: List of model results dictionaries
            features: Feature matrix for computing metrics
            
        Returns:
            DataFrame with comparison metrics
        """
        if len(results_list) < 2:
            st.warning("Need at least 2 models for comparison")
            return pd.DataFrame()
        
        comparison_data = []
        
        for i, results in enumerate(results_list):
            model_name = f"Model {i+1}"
            metrics = self._compute_model_metrics(results, features, model_name)
            comparison_data.append(metrics)
        
        return pd.DataFrame(comparison_data)
    
    def _compute_model_metrics(self, results: Dict[str, Any], 
                              features: Optional[pd.DataFrame], 
                              model_name: str) -> Dict[str, Any]:
        """Compute metrics for a single model."""
        metrics = {"Model": model_name}
        
        # Basic statistics
        if "labels" in results:
            labels = results["labels"]
            if isinstance(labels, pd.DataFrame) and "regime" in labels.columns:
                regime_labels = labels["regime"]
            else:
                regime_labels = labels
            
            stats = self.utils.calculate_regime_statistics(regime_labels)
            
            metrics.update({
                "Total_Observations": stats.get("total_observations", 0),
                "Unique_Regimes": stats.get("unique_regimes", 0),
                "Regime_Diversity": stats.get("regime_diversity", 0),
                "Most_Common_Regime": stats.get("most_common_regime", "N/A")
            })
            
            # Clustering quality metrics (if features available)
            if features is not None and len(regime_labels) == len(features):
                try:
                    # Align features and labels
                    aligned_features = features.iloc[:len(regime_labels)]
                    aligned_labels = regime_labels.iloc[:len(aligned_features)]
                    
                    # Remove any NaN values
                    mask = ~(aligned_labels.isna() | aligned_features.isna().any(axis=1))
                    clean_features = aligned_features[mask]
                    clean_labels = aligned_labels[mask]
                    
                    if len(clean_features) > 0 and len(clean_labels) > 0:
                        # Silhouette score
                        try:
                            silhouette = silhouette_score(clean_features, clean_labels)
                            metrics["Silhouette_Score"] = silhouette
                        except:
                            metrics["Silhouette_Score"] = np.nan
                        
                        # Calinski-Harabasz score
                        try:
                            ch_score = calinski_harabasz_score(clean_features, clean_labels)
                            metrics["Calinski_Harabasz_Score"] = ch_score
                        except:
                            metrics["Calinski_Harabasz_Score"] = np.nan
                        
                        # Davies-Bouldin score
                        try:
                            db_score = davies_bouldin_score(clean_features, clean_labels)
                            metrics["Davies_Bouldin_Score"] = db_score
                        except:
                            metrics["Davies_Bouldin_Score"] = np.nan
                    
                except Exception as e:
                    st.warning(f"Error computing clustering metrics: {e}")
            
            # Regime stability metrics
            stability_metrics = self._compute_regime_stability(regime_labels)
            metrics.update(stability_metrics)
        
        # Execution information
        if "execution_info" in results:
            exec_info = results["execution_info"]
            metrics.update({
                "Algorithm": exec_info.get("algorithm", "N/A"),
                "Mode": exec_info.get("mode", "N/A"),
                "Execution_Time": exec_info.get("timestamp", "N/A")
            })
        
        return metrics
    
    def _compute_regime_stability(self, regime_labels: pd.Series) -> Dict[str, float]:
        """Compute regime stability metrics."""
        if len(regime_labels) < 2:
            return {"Regime_Stability": np.nan, "Transition_Rate": np.nan}
        
        # Count transitions
        transitions = 0
        for i in range(1, len(regime_labels)):
            if regime_labels.iloc[i] != regime_labels.iloc[i-1]:
                transitions += 1
        
        # Transition rate
        transition_rate = transitions / (len(regime_labels) - 1)
        
        # Regime stability (inverse of transition rate)
        stability = 1 - transition_rate
        
        # Average regime duration
        regime_durations = []
        current_regime = regime_labels.iloc[0]
        current_duration = 1
        
        for i in range(1, len(regime_labels)):
            if regime_labels.iloc[i] == current_regime:
                current_duration += 1
            else:
                regime_durations.append(current_duration)
                current_regime = regime_labels.iloc[i]
                current_duration = 1
        
        regime_durations.append(current_duration)  # Add last regime duration
        avg_duration = np.mean(regime_durations) if regime_durations else 0
        
        return {
            "Regime_Stability": stability,
            "Transition_Rate": transition_rate,
            "Avg_Regime_Duration": avg_duration
        }
    
    def display_comparison_table(self, comparison_df: pd.DataFrame) -> None:
        """Display comparison table with formatting."""
        if comparison_df.empty:
            st.warning("No comparison data available")
            return
        
        st.subheader("📊 Model Comparison Table")
        
        # Format numeric columns
        numeric_cols = comparison_df.select_dtypes(include=[np.number]).columns
        formatted_df = comparison_df.copy()
        
        for col in numeric_cols:
            if col in ["Silhouette_Score", "Calinski_Harabasz_Score", "Davies_Bouldin_Score"]:
                formatted_df[col] = formatted_df[col].round(3)
            elif col in ["Regime_Stability", "Transition_Rate"]:
                formatted_df[col] = formatted_df[col].round(3)
            elif col in ["Avg_Regime_Duration"]:
                formatted_df[col] = formatted_df[col].round(1)
        
        st.dataframe(formatted_df, use_container_width=True)
    
    def display_comparison_charts(self, comparison_df: pd.DataFrame) -> None:
        """Display comparison charts."""
        if comparison_df.empty:
            return
        
        st.subheader("📈 Comparison Charts")
        
        # Select metrics to plot
        numeric_cols = comparison_df.select_dtypes(include=[np.number]).columns
        plot_metrics = [col for col in numeric_cols if col not in ["Total_Observations", "Unique_Regimes"]]
        
        if not plot_metrics:
            st.info("No numeric metrics available for plotting")
            return
        
        # Create subplots
        n_metrics = len(plot_metrics)
        n_cols = min(2, n_metrics)
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))
        if n_metrics == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes
        else:
            axes = axes.flatten()
        
        for i, metric in enumerate(plot_metrics):
            if i >= len(axes):
                break
                
            ax = axes[i]
            
            # Bar plot for each metric
            bars = ax.bar(comparison_df["Model"], comparison_df[metric], 
                         color=plt.cm.Set3(np.linspace(0, 1, len(comparison_df))))
            
            ax.set_title(f"{metric.replace('_', ' ').title()}")
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                if not np.isnan(height):
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.3f}', ha='center', va='bottom')
        
        # Hide unused subplots
        for i in range(len(plot_metrics), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    def display_ranking(self, comparison_df: pd.DataFrame) -> None:
        """Display model ranking based on different metrics."""
        if comparison_df.empty:
            return
        
        st.subheader("🏆 Model Rankings")
        
        # Define ranking criteria (higher is better for most metrics)
        ranking_criteria = {
            "Silhouette_Score": "desc",
            "Calinski_Harabasz_Score": "desc", 
            "Davies_Bouldin_Score": "asc",  # Lower is better
            "Regime_Stability": "desc",
            "Avg_Regime_Duration": "desc"
        }
        
        rankings = {}
        
        for metric, direction in ranking_criteria.items():
            if metric in comparison_df.columns:
                # Remove NaN values for ranking
                valid_data = comparison_df[["Model", metric]].dropna()
                if len(valid_data) > 1:
                    if direction == "desc":
                        ranked = valid_data.sort_values(metric, ascending=False)
                    else:
                        ranked = valid_data.sort_values(metric, ascending=True)
                    
                    rankings[metric] = ranked["Model"].tolist()
        
        if rankings:
            # Create ranking table
            ranking_df = pd.DataFrame(rankings)
            ranking_df.index = range(1, len(ranking_df) + 1)
            ranking_df.index.name = "Rank"
            
            st.dataframe(ranking_df, use_container_width=True)
            
            # Overall ranking (simple average of ranks)
            model_ranks = {}
            for model in comparison_df["Model"]:
                ranks = []
                for metric, ranking in rankings.items():
                    if model in ranking:
                        ranks.append(ranking.index(model) + 1)
                
                if ranks:
                    model_ranks[model] = np.mean(ranks)
            
            if model_ranks:
                overall_ranking = sorted(model_ranks.items(), key=lambda x: x[1])
                
                st.subheader("🥇 Overall Ranking")
                for i, (model, avg_rank) in enumerate(overall_ranking, 1):
                    st.write(f"{i}. **{model}** (Average Rank: {avg_rank:.1f})")
        else:
            st.info("No valid metrics available for ranking")
    
    def generate_comparison_report(self, comparison_df: pd.DataFrame) -> str:
        """Generate a text report of the comparison."""
        if comparison_df.empty:
            return "No comparison data available."
        
        report = ["# Model Comparison Report\n"]
        
        # Summary statistics
        report.append("## Summary Statistics\n")
        for _, row in comparison_df.iterrows():
            report.append(f"### {row['Model']}")
            report.append(f"- Algorithm: {row.get('Algorithm', 'N/A')}")
            report.append(f"- Mode: {row.get('Mode', 'N/A')}")
            report.append(f"- Total Observations: {row.get('Total_Observations', 'N/A')}")
            report.append(f"- Unique Regimes: {row.get('Unique_Regimes', 'N/A')}")
            report.append(f"- Regime Diversity: {row.get('Regime_Diversity', 'N/A'):.3f}")
            report.append("")
        
        # Performance metrics
        numeric_cols = comparison_df.select_dtypes(include=[np.number]).columns
        performance_metrics = [col for col in numeric_cols if col in [
            "Silhouette_Score", "Calinski_Harabasz_Score", "Davies_Bouldin_Score",
            "Regime_Stability", "Transition_Rate", "Avg_Regime_Duration"
        ]]
        
        if performance_metrics:
            report.append("## Performance Metrics\n")
            for metric in performance_metrics:
                valid_data = comparison_df[["Model", metric]].dropna()
                if len(valid_data) > 0:
                    best_model = valid_data.loc[valid_data[metric].idxmax(), "Model"]
                    best_value = valid_data[metric].max()
                    report.append(f"- **{metric}**: {best_model} ({best_value:.3f})")
            report.append("")
        
        return "\n".join(report)
    
    def export_comparison_data(self, comparison_df: pd.DataFrame, format: str = "csv") -> str:
        """Export comparison data in specified format."""
        if format.lower() == "csv":
            return comparison_df.to_csv(index=False)
        elif format.lower() == "json":
            return comparison_df.to_json(orient="records", indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")
