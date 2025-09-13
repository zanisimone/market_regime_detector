"""
Results display component for clustering outputs.

This module provides comprehensive visualization and display of clustering results
with interactive charts and detailed analysis.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, List, Optional
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from .utils import ClusteringUtils


class ResultsDisplay:
    """Component for displaying clustering results."""
    
    def __init__(self):
        self.utils = ClusteringUtils()
    
    def display_results_overview(self, results: Dict[str, Any]) -> None:
        """Display overview of clustering results."""
        st.subheader("📊 Results Overview")
        
        if not results:
            st.warning("No results to display")
            return
        
        # Basic statistics
        if "labels" in results:
            labels = results["labels"]
            if isinstance(labels, pd.DataFrame) and "regime" in labels.columns:
                regime_labels = labels["regime"]
            else:
                regime_labels = labels
            
            stats = self.utils.calculate_regime_statistics(regime_labels)
            
            # Display key metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Observations", stats.get("total_observations", 0))
            
            with col2:
                st.metric("Unique Regimes", stats.get("unique_regimes", 0))
            
            with col3:
                most_common = stats.get("most_common_regime", "N/A")
                st.metric("Most Common Regime", most_common)
            
            with col4:
                diversity = stats.get("regime_diversity", 0)
                st.metric("Regime Diversity", f"{diversity:.2f}")
        
        # Regime distribution
        if "labels" in results:
            self._display_regime_distribution(results["labels"])
    
    def display_regime_analysis(self, results: Dict[str, Any]) -> None:
        """Display detailed regime analysis."""
        st.subheader("🔍 Regime Analysis")
        
        if "labels" not in results:
            st.warning("No regime labels found in results")
            return
        
        labels = results["labels"]
        
        # Extract regime column
        if isinstance(labels, pd.DataFrame):
            if "regime" in labels.columns:
                regime_col = labels["regime"]
                date_col = labels.index if hasattr(labels, 'index') else None
            else:
                st.error("No 'regime' column found in labels")
                return
        else:
            regime_col = labels
            date_col = None
        
        # Regime timeline
        self._display_regime_timeline(regime_col, date_col)
        
        # Regime statistics table
        self._display_regime_statistics_table(regime_col)
        
        # Regime transitions (if available)
        if date_col is not None:
            self._display_regime_transitions(regime_col, date_col)
    
    def display_performance_metrics(self, results: Dict[str, Any]) -> None:
        """Display performance metrics and evaluation."""
        st.subheader("📈 Performance Metrics")
        
        # Algorithm-specific metrics
        if "centers" in results or "means" in results:
            self._display_cluster_centers(results)
        
        # Model quality metrics (if available)
        if "model" in results:
            self._display_model_quality(results["model"])
        
        # Execution information
        if "execution_info" in results:
            self._display_execution_info(results["execution_info"])
    
    def display_model_comparison(self, results_list: List[Dict[str, Any]]) -> None:
        """Display comparison between multiple models."""
        st.subheader("⚖️ Model Comparison")
        
        if len(results_list) < 2:
            st.info("Need at least 2 models for comparison")
            return
        
        # Create comparison table
        comparison_data = []
        for i, results in enumerate(results_list):
            if "labels" in results:
                labels = results["labels"]
                if isinstance(labels, pd.DataFrame) and "regime" in labels.columns:
                    regime_labels = labels["regime"]
                else:
                    regime_labels = labels
                
                stats = self.utils.calculate_regime_statistics(regime_labels)
                
                comparison_data.append({
                    "Model": f"Model {i+1}",
                    "Observations": stats.get("total_observations", 0),
                    "Regimes": stats.get("unique_regimes", 0),
                    "Diversity": f"{stats.get('regime_diversity', 0):.4f}",
                    "Most Common": stats.get("most_common_regime", "N/A")
                })
        
        if comparison_data:
            df_comparison = pd.DataFrame(comparison_data)
            st.dataframe(df_comparison, use_container_width=True)
    
    def _display_regime_distribution(self, labels: Any) -> None:
        """Display regime distribution chart."""
        if isinstance(labels, pd.DataFrame) and "regime" in labels.columns:
            regime_col = labels["regime"]
        else:
            regime_col = labels
        
        # Count regimes
        regime_counts = regime_col.value_counts().sort_index()
        
        # Create visualization
        col1, col2 = st.columns(2)
        
        with col1:
            # Bar chart
            fig, ax = plt.subplots(figsize=(8, 4))
            regime_counts.plot(kind='bar', ax=ax, color='skyblue', edgecolor='black')
            ax.set_title('Regime Distribution')
            ax.set_xlabel('Regime')
            ax.set_ylabel('Count')
            ax.tick_params(axis='x', rotation=0)
            st.pyplot(fig)
            plt.close()
        
        with col2:
            # Pie chart
            fig, ax = plt.subplots(figsize=(8, 4))
            regime_counts.plot(kind='pie', ax=ax, autopct='%1.1f%%', startangle=90)
            ax.set_title('Regime Proportions')
            ax.set_ylabel('')
            st.pyplot(fig)
            plt.close()
    
    def _display_regime_timeline(self, regime_col: pd.Series, date_col: Optional[pd.Index]) -> None:
        """Display regime timeline."""
        if date_col is None:
            st.warning("No date information available for timeline")
            return
        
        # Create timeline plot
        fig, ax = plt.subplots(figsize=(12, 4))
        
        # Plot regime sequence
        ax.plot(date_col, regime_col, marker='o', markersize=1, linewidth=0.5)
        ax.set_ylabel('Regime')
        ax.set_xlabel('Date')
        ax.set_title('Regime Timeline')
        ax.grid(True, alpha=0.3)
        
        # Color code regimes
        unique_regimes = regime_col.unique()
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_regimes)))
        
        for i, regime in enumerate(unique_regimes):
            mask = regime_col == regime
            ax.scatter(date_col[mask], regime_col[mask], 
                      c=[colors[i]], label=f'Regime {regime}', s=10, alpha=0.7)
        
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    def _display_regime_statistics_table(self, regime_col: pd.Series) -> None:
        """Display regime statistics table."""
        regime_counts = regime_col.value_counts().sort_index()
        regime_props = (regime_counts / len(regime_col) * 100).round(2)
        
        stats_df = pd.DataFrame({
            'Regime': regime_counts.index,
            'Count': regime_counts.values,
            'Percentage': regime_props.values
        })
        
        st.dataframe(stats_df, use_container_width=True)
    
    def _display_regime_transitions(self, regime_col: pd.Series, date_col: pd.Index) -> None:
        """Display regime transition analysis."""
        st.subheader("🔄 Regime Transitions")
        
        # Calculate transition matrix
        transitions = []
        for i in range(1, len(regime_col)):
            from_regime = regime_col.iloc[i-1]
            to_regime = regime_col.iloc[i]
            if from_regime != to_regime:  # Only count actual transitions
                transitions.append((from_regime, to_regime))
        
        if transitions:
            # Create transition matrix
            unique_regimes = sorted(regime_col.unique())
            transition_matrix = pd.DataFrame(
                index=unique_regimes, 
                columns=unique_regimes, 
                dtype=float
            ).fillna(0)
            
            for from_regime, to_regime in transitions:
                transition_matrix.loc[from_regime, to_regime] += 1
            
            # Normalize by row sums
            row_sums = transition_matrix.sum(axis=1)
            transition_matrix = transition_matrix.div(row_sums, axis=0).fillna(0)
            
            # Display heatmap
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(transition_matrix, annot=True, fmt='.2f', cmap='Blues', ax=ax)
            ax.set_title('Regime Transition Matrix')
            ax.set_xlabel('To Regime')
            ax.set_ylabel('From Regime')
            st.pyplot(fig)
            plt.close()
            
            # Display transition statistics
            st.write(f"**Total Transitions:** {len(transitions)}")
            st.write(f"**Average Regime Duration:** {len(regime_col) / len(transitions):.1f} periods")
        else:
            st.info("No regime transitions detected")
    
    def _display_cluster_centers(self, results: Dict[str, Any]) -> None:
        """Display cluster centers or means."""
        st.subheader("🎯 Cluster Centers")
        
        centers_key = "centers" if "centers" in results else "means"
        if centers_key not in results:
            return
        
        centers = results[centers_key]
        
        if isinstance(centers, pd.DataFrame):
            st.dataframe(centers, use_container_width=True)
            
            # Visualize centers if we have feature names
            if len(centers.columns) > 1:
                self._plot_cluster_centers(centers)
        else:
            st.write("Cluster centers data format not recognized")
    
    def _plot_cluster_centers(self, centers: pd.DataFrame) -> None:
        """Plot cluster centers visualization."""
        if len(centers.columns) < 2:
            return
        
        # Use first two features for 2D plot
        feature1, feature2 = centers.columns[:2]
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        for i, (idx, row) in enumerate(centers.iterrows()):
            ax.scatter(row[feature1], row[feature2], 
                      label=f'Cluster {idx}', s=100, alpha=0.7)
            ax.annotate(f'C{idx}', (row[feature1], row[feature2]), 
                       xytext=(5, 5), textcoords='offset points')
        
        ax.set_xlabel(feature1)
        ax.set_ylabel(feature2)
        ax.set_title('Cluster Centers (2D Projection)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        st.pyplot(fig)
        plt.close()
    
    def _display_model_quality(self, model: Any) -> None:
        """Display model quality metrics."""
        st.subheader("🔬 Model Quality")
        
        # This would need to be implemented based on the specific model type
        # For now, just show that we have a model
        st.info("Model object loaded successfully")
        
        # Try to extract some basic information
        if hasattr(model, 'inertia_'):
            st.metric("Inertia", f"{model.inertia_:.2f}")
        
        if hasattr(model, 'n_iter_'):
            st.metric("Iterations", model.n_iter_)
    
    def _display_execution_info(self, execution_info: Dict[str, Any]) -> None:
        """Display execution information."""
        st.subheader("ℹ️ Execution Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**Algorithm:** {execution_info.get('algorithm', 'N/A')}")
            st.write(f"**Mode:** {execution_info.get('mode', 'N/A')}")
            st.write(f"**Output Directory:** {execution_info.get('output_dir', 'N/A')}")
        
        with col2:
            timestamp = execution_info.get('timestamp', 0)
            if timestamp:
                from datetime import datetime
                dt = datetime.fromtimestamp(timestamp)
                st.write(f"**Execution Time:** {dt.strftime('%Y-%m-%d %H:%M:%S')}")
            
            st.write(f"**Command:** `{execution_info.get('command', 'N/A')}`")
    
    def create_download_links(self, results: Dict[str, Any], algorithm: str, mode: str) -> None:
        """Create download links for results."""
        st.subheader("💾 Download Results")
        
        if not results:
            st.warning("No results to download")
            return
        
        # Create download links for each result type
        for result_type, data in results.items():
            if result_type == "execution_info":
                continue  # Skip execution info
            
            if isinstance(data, pd.DataFrame):
                filename = f"{algorithm}_{mode}_{result_type}"
                csv = data.to_csv(index=True)
                
                st.download_button(
                    label=f"Download {result_type.title()}",
                    data=csv,
                    file_name=f"{filename}.csv",
                    mime="text/csv"
                )
