"""
Algorithm selection component for clustering interface.

This module provides a user-friendly interface for selecting clustering algorithms
and their associated parameters and modes.
"""

import streamlit as st
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from .validation import ParameterValidator


class AlgorithmSelector:
    """Component for selecting and configuring clustering algorithms."""
    
    def __init__(self):
        self.validator = ParameterValidator()
        self.algorithms_config = self._load_algorithms_config()
    
    def _load_algorithms_config(self) -> Dict:
        """Load algorithms configuration from JSON file."""
        try:
            config_path = Path(__file__).parent.parent.parent / "configs" / "algorithms.json"
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            st.error(f"Error loading algorithms configuration: {e}")
            return {}
    
    def render_algorithm_selection(self) -> Tuple[Optional[str], Optional[str]]:
        """
        Render algorithm and mode selection interface.
        
        Returns:
            Tuple of (selected_algorithm, selected_mode)
        """
        st.subheader("🔧 Algorithm Selection")
        
        # Algorithm selection
        algorithm_options = list(self.algorithms_config.keys())
        algorithm_names = [self.algorithms_config[algo]["name"] for algo in algorithm_options]
        
        selected_algorithm_idx = st.selectbox(
            "Select Algorithm",
            range(len(algorithm_options)),
            format_func=lambda x: algorithm_names[x],
            help="Choose the clustering algorithm to use"
        )
        
        selected_algorithm = algorithm_options[selected_algorithm_idx] if selected_algorithm_idx is not None else None
        
        if selected_algorithm:
            # Display algorithm information
            self._display_algorithm_info(selected_algorithm)
            
            # Mode selection
            selected_mode = self._render_mode_selection(selected_algorithm)
            
            return selected_algorithm, selected_mode
        
        return None, None
    
    def _display_algorithm_info(self, algorithm: str) -> None:
        """Display detailed information about the selected algorithm."""
        algo_config = self.algorithms_config[algorithm]
        
        # Create columns for info display
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown(f"**{algo_config['name']}**")
            st.markdown(f"*{algo_config['description']}*")
        
        with col2:
            if st.button("ℹ️ Details", key=f"details_{algorithm}"):
                self._show_algorithm_details(algorithm)
    
    def _show_algorithm_details(self, algorithm: str) -> None:
        """Show detailed algorithm information in an expander."""
        algo_config = self.algorithms_config[algorithm]
        
        with st.expander(f"📋 {algo_config['name']} Details", expanded=True):
            st.markdown("**Description:**")
            st.markdown(algo_config['description'])
            
            st.markdown("**Advantages:**")
            for pro in algo_config.get('pros', []):
                st.markdown(f"• {pro}")
            
            st.markdown("**Limitations:**")
            for con in algo_config.get('cons', []):
                st.markdown(f"• {con}")
            
            st.markdown("**Supported Modes:**")
            for mode in algo_config.get('modes', []):
                st.markdown(f"• {mode.title()}")
    
    def _render_mode_selection(self, algorithm: str) -> Optional[str]:
        """Render training mode selection for the algorithm."""
        algo_config = self.algorithms_config[algorithm]
        supported_modes = algo_config.get('modes', [])
        
        if len(supported_modes) == 1:
            # Only one mode supported, select it automatically
            selected_mode = supported_modes[0]
            st.info(f"Mode: **{selected_mode.title()}** (only mode supported)")
            return selected_mode
        
        # Multiple modes available, let user choose
        mode_descriptions = {
            'split': 'Train once on historical data, apply to full dataset',
            'rolling': 'Retrain on moving window for time-adaptive regimes'
        }
        
        selected_mode = st.radio(
            "Training Mode",
            supported_modes,
            format_func=lambda x: f"{x.title()}: {mode_descriptions.get(x, '')}",
            help="Choose how the algorithm should be trained"
        )
        
        return selected_mode
    
    def get_algorithm_parameters(self, algorithm: str) -> Dict:
        """Get parameter configuration for the selected algorithm."""
        if algorithm not in self.algorithms_config:
            return {}
        
        return self.algorithms_config[algorithm].get('parameters', {})
    
    def get_algorithm_script(self, algorithm: str, mode: str) -> Optional[str]:
        """Get the script name for the selected algorithm and mode."""
        if algorithm not in self.algorithms_config:
            return None
        
        algo_config = self.algorithms_config[algorithm]
        
        if mode == 'rolling':
            return algo_config.get('rolling_script')
        else:
            return algo_config.get('script')
    
    def get_output_files(self, algorithm: str) -> Dict[str, str]:
        """Get expected output file names for the algorithm."""
        if algorithm not in self.algorithms_config:
            return {}
        
        return self.algorithms_config[algorithm].get('output_files', {})
    
    def validate_algorithm_selection(self, algorithm: str, mode: str) -> bool:
        """Validate the algorithm and mode selection."""
        if not algorithm or not mode:
            return False
        
        is_valid, error_msg = self.validator.validate_algorithm_mode_combination(algorithm, mode)
        
        if not is_valid:
            st.error(f"Validation Error: {error_msg}")
            return False
        
        return True
    
    def render_algorithm_comparison(self) -> None:
        """Render a comparison table of all available algorithms."""
        st.subheader("📊 Algorithm Comparison")
        
        if not self.algorithms_config:
            st.warning("No algorithm configurations available")
            return
        
        # Create comparison data
        comparison_data = []
        for algo_key, algo_config in self.algorithms_config.items():
            comparison_data.append({
                'Algorithm': algo_config['name'],
                'Type': self._get_algorithm_type(algo_key),
                'Modes': ', '.join([m.title() for m in algo_config.get('modes', [])]),
                'Complexity': self._get_complexity_level(algo_key),
                'Best For': self._get_best_use_case(algo_key)
            })
        
        # Display comparison table
        import pandas as pd
        df = pd.DataFrame(comparison_data)
        st.dataframe(df, use_container_width=True)
    
    def _get_algorithm_type(self, algorithm: str) -> str:
        """Get algorithm type description."""
        type_map = {
            'kmeans': 'Hard Clustering',
            'gmm': 'Soft Clustering',
            'hmm': 'Sequential Modeling'
        }
        return type_map.get(algorithm, 'Unknown')
    
    def _get_complexity_level(self, algorithm: str) -> str:
        """Get complexity level description."""
        complexity_map = {
            'kmeans': 'Low',
            'gmm': 'Medium',
            'hmm': 'High'
        }
        return complexity_map.get(algorithm, 'Unknown')
    
    def _get_best_use_case(self, algorithm: str) -> str:
        """Get best use case description."""
        use_case_map = {
            'kmeans': 'Quick regime identification',
            'gmm': 'Uncertainty-aware clustering',
            'hmm': 'Temporal regime analysis'
        }
        return use_case_map.get(algorithm, 'Unknown')
    
    def get_algorithm_recommendations(self, use_case: str) -> List[str]:
        """
        Get algorithm recommendations based on use case.
        
        Args:
            use_case: Description of the intended use case
            
        Returns:
            List of recommended algorithm keys
        """
        recommendations = {
            'quick_analysis': ['kmeans'],
            'uncertainty_important': ['gmm'],
            'temporal_patterns': ['hmm'],
            'comprehensive_analysis': ['kmeans', 'gmm', 'hmm'],
            'production_system': ['kmeans', 'gmm'],
            'research_analysis': ['hmm', 'gmm']
        }
        
        # Simple keyword matching for use case
        use_case_lower = use_case.lower()
        
        if any(word in use_case_lower for word in ['quick', 'fast', 'simple']):
            return recommendations['quick_analysis']
        elif any(word in use_case_lower for word in ['uncertainty', 'probability', 'confidence']):
            return recommendations['uncertainty_important']
        elif any(word in use_case_lower for word in ['temporal', 'time', 'sequence', 'persistence']):
            return recommendations['temporal_patterns']
        elif any(word in use_case_lower for word in ['comprehensive', 'complete', 'thorough']):
            return recommendations['comprehensive_analysis']
        elif any(word in use_case_lower for word in ['production', 'live', 'real-time']):
            return recommendations['production_system']
        else:
            return recommendations['research_analysis']
