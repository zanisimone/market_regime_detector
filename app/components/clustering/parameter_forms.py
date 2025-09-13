"""
Dynamic parameter forms for clustering algorithms.

This module provides dynamic form generation based on algorithm configuration,
with real-time validation and user-friendly interfaces.
"""

import streamlit as st
from typing import Dict, Any, List, Optional, Tuple
from .validation import ParameterValidator, ValidationResult
import pandas as pd


class ParameterForm:
    """Dynamic parameter form generator for clustering algorithms."""
    
    def __init__(self):
        self.validator = ParameterValidator()
    
    def render_parameter_form(self, algorithm: str, mode: str, 
                            current_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Render a dynamic parameter form for the selected algorithm.
        
        Args:
            algorithm: Selected algorithm name
            mode: Selected training mode
            current_params: Current parameter values (for editing)
            
        Returns:
            Dictionary of parameter values
        """
        st.subheader("⚙️ Algorithm Parameters")
        
        # Load algorithm configuration
        algorithm_config = self._get_algorithm_config(algorithm)
        if not algorithm_config:
            st.error(f"Configuration not found for algorithm: {algorithm}")
            return {}
        
        parameters_config = algorithm_config.get('parameters', {})
        if not parameters_config:
            st.info("No parameters required for this algorithm")
            return {}
        
        # Create form
        form_key = f"param_form_{algorithm}_{mode}"
        with st.form(key=form_key):
            params = {}
            
            # Render each parameter
            for param_name, param_config in parameters_config.items():
                param_value = self._render_parameter_input(
                    param_name, param_config, 
                    current_params.get(param_name) if current_params else None
                )
                params[param_name] = param_value
            
            # Add mode-specific parameters
            if mode == "rolling":
                rolling_params = self._render_rolling_parameters()
                params.update(rolling_params)
            
            # Add date range parameters for split mode
            if mode == "split":
                date_params = self._render_date_parameters()
                params.update(date_params)
            
            # Form submission
            submitted = st.form_submit_button(
                "🔍 Validate Parameters",
                help="Validate parameters before running the algorithm"
            )
            
            if submitted:
                validation_result = self._validate_parameters(algorithm, mode, params)
                validation_result.display_in_streamlit()
                
                if validation_result.is_valid:
                    st.session_state[f"validated_params_{algorithm}_{mode}"] = params
                    st.session_state[f"validation_timestamp_{algorithm}_{mode}"] = st.session_state.get("form_submit_count", 0)
        
        # Return current parameters (from session state if validated)
        validated_params = st.session_state.get(f"validated_params_{algorithm}_{mode}", {})
        return validated_params if validated_params else params
    
    def _get_algorithm_config(self, algorithm: str) -> Optional[Dict]:
        """Get algorithm configuration."""
        try:
            import json
            from pathlib import Path
            config_path = Path(__file__).parent.parent.parent / "configs" / "algorithms.json"
            with open(config_path, 'r') as f:
                algorithms_config = json.load(f)
            return algorithms_config.get(algorithm)
        except Exception as e:
            st.error(f"Error loading algorithm configuration: {e}")
            return None
    
    def _render_parameter_input(self, param_name: str, param_config: Dict, 
                              current_value: Any = None) -> Any:
        """Render a single parameter input based on its configuration."""
        param_type = param_config.get('type', 'text')
        param_help = param_config.get('help', '')
        param_default = param_config.get('default')
        
        # Use current value or default
        initial_value = current_value if current_value is not None else param_default
        
        # Create label with help
        label = param_name.replace('_', ' ').title()
        if param_help:
            label += f" ({param_help})"
        
        if param_type == "int":
            min_val = param_config.get('min', 0)
            max_val = param_config.get('max', 1000)
            step = param_config.get('step', 1)
            
            return st.number_input(
                label,
                min_value=min_val,
                max_value=max_val,
                value=initial_value,
                step=step,
                help=param_help,
                key=f"param_{param_name}"
            )
        
        elif param_type == "float":
            min_val = param_config.get('min', 0.0)
            max_val = param_config.get('max', 1.0)
            step = param_config.get('step', 0.01)
            
            return st.number_input(
                label,
                min_value=min_val,
                max_value=max_val,
                value=float(initial_value) if initial_value is not None else param_default,
                step=step,
                help=param_help,
                key=f"param_{param_name}"
            )
        
        elif param_type == "select":
            options = param_config.get('options', [])
            option_labels = [opt.replace('_', ' ').title() for opt in options]
            
            selected_idx = 0
            if initial_value in options:
                selected_idx = options.index(initial_value)
            
            selected_label = st.selectbox(
                label,
                range(len(options)),
                index=selected_idx,
                format_func=lambda x: option_labels[x],
                help=param_help,
                key=f"param_{param_name}"
            )
            
            return options[selected_label]
        
        elif param_type == "bool":
            return st.checkbox(
                label,
                value=initial_value if initial_value is not None else param_default,
                help=param_help,
                key=f"param_{param_name}"
            )
        
        else:  # text or unknown type
            return st.text_input(
                label,
                value=str(initial_value) if initial_value is not None else str(param_default),
                help=param_help,
                key=f"param_{param_name}"
            )
    
    def _render_rolling_parameters(self) -> Dict[str, Any]:
        """Render rolling-specific parameters."""
        st.markdown("**Rolling Window Parameters**")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            start_date = st.date_input(
                "Start Date",
                value=pd.Timestamp("2010-01-01").date(),
                min_value=pd.Timestamp("2000-01-01").date(),
                max_value=pd.Timestamp("2030-12-31").date(),
                help="Start date for rolling backtest"
            )
            end_date = st.date_input(
                "End Date",
                value=pd.Timestamp("2024-12-31").date(),
                min_value=pd.Timestamp("2000-01-01").date(),
                max_value=pd.Timestamp("2030-12-31").date(),
                help="End date for rolling backtest"
            )
        
        with col2:
            lookback_days = st.number_input(
                "Lookback Days",
                min_value=30,
                max_value=1000,
                value=504,
                help="Number of days for training window"
            )
            step_days = st.number_input(
                "Step Days",
                min_value=1,
                max_value=100,
                value=21,
                help="Number of days between refits"
            )
        
        with col3:
            oos_days = st.number_input(
                "Out-of-Sample Days",
                min_value=1,
                max_value=100,
                value=21,
                help="Number of days for out-of-sample prediction"
            )
        
        return {
            "start": start_date.strftime("%Y-%m-%d"),
            "end": end_date.strftime("%Y-%m-%d"),
            "lookback_days": lookback_days,
            "step_days": step_days,
            "oos_days": oos_days
        }
    
    def _render_date_parameters(self) -> Dict[str, Any]:
        """Render date range parameters for split mode."""
        st.markdown("**Date Range Parameters**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            train_start = st.date_input(
                "Training Start Date",
                value=pd.Timestamp("2010-01-01").date(),
                min_value=pd.Timestamp("2000-01-01").date(),
                max_value=pd.Timestamp("2024-12-31").date(),
                help="Start date for training period"
            )
            train_end = st.date_input(
                "Training End Date", 
                value=pd.Timestamp("2018-12-31").date(),
                min_value=pd.Timestamp("2000-01-01").date(),
                max_value=pd.Timestamp("2024-12-31").date(),
                help="End date for training period"
            )
        
        with col2:
            test_start = st.date_input(
                "Test Start Date",
                value=pd.Timestamp("2019-01-01").date(),
                min_value=pd.Timestamp("2000-01-01").date(),
                max_value=pd.Timestamp("2024-12-31").date(),
                help="Start date for test period"
            )
            test_end = st.date_input(
                "Test End Date",
                value=pd.Timestamp("2024-12-31").date(),
                min_value=pd.Timestamp("2000-01-01").date(),
                max_value=pd.Timestamp("2024-12-31").date(),
                help="End date for test period"
            )
        
        # Convert dates to strings
        date_params = {}
        if train_start:
            date_params["train_start"] = train_start.strftime("%Y-%m-%d")
        if train_end:
            date_params["train_end"] = train_end.strftime("%Y-%m-%d")
        if test_start:
            date_params["test_start"] = test_start.strftime("%Y-%m-%d")
        if test_end:
            date_params["test_end"] = test_end.strftime("%Y-%m-%d")
        
        return date_params
    
    def _validate_parameters(self, algorithm: str, mode: str, params: Dict[str, Any]) -> ValidationResult:
        """Validate all parameters for the algorithm and mode."""
        result = ValidationResult(is_valid=True)
        
        # Validate algorithm-specific parameters
        is_valid, errors = self.validator.validate_algorithm_params(algorithm, params)
        if not is_valid:
            for error in errors:
                result.add_error(error)
        
        # Validate date ranges for split mode
        if mode == "split":
            date_fields = ["train_start", "train_end", "test_start", "test_end"]
            date_values = {field: params.get(field) for field in date_fields if params.get(field)}
            
            if len(date_values) == 4:
                is_valid, errors = self.validator.validate_date_ranges(
                    date_values["train_start"],
                    date_values["train_end"],
                    date_values["test_start"],
                    date_values["test_end"]
                )
                if not is_valid:
                    for error in errors:
                        result.add_error(error)
            else:
                result.add_error("All date fields are required for split mode")
        
        # Validate rolling parameters for rolling mode
        if mode == "rolling":
            window_size = params.get("window_size", 252)
            step_size = params.get("step_size", 1)
            
            is_valid, errors = self.validator.validate_rolling_params(window_size, step_size)
            if not is_valid:
                for error in errors:
                    result.add_error(error)
        
        return result
    
    def render_parameter_summary(self, algorithm: str, mode: str, params: Dict[str, Any]) -> None:
        """Render a summary of the current parameters."""
        if not params:
            st.info("No parameters configured")
            return
        
        st.subheader("📋 Parameter Summary")
        
        # Create a nice display of parameters
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Algorithm Parameters:**")
            for param_name, param_value in params.items():
                if param_name not in ["train_start", "train_end", "test_start", "test_end", "window_size", "step_size"]:
                    st.markdown(f"• **{param_name.replace('_', ' ').title()}**: {param_value}")
        
        with col2:
            if mode == "split":
                st.markdown("**Date Range:**")
                if "train_start" in params:
                    st.markdown(f"• **Training**: {params['train_start']} to {params.get('train_end', 'N/A')}")
                if "test_start" in params:
                    st.markdown(f"• **Test**: {params['test_start']} to {params.get('test_end', 'N/A')}")
            
            elif mode == "rolling":
                st.markdown("**Rolling Parameters:**")
                if "window_size" in params:
                    st.markdown(f"• **Window Size**: {params['window_size']}")
                if "step_size" in params:
                    st.markdown(f"• **Step Size**: {params['step_size']}")
    
    def get_default_parameters(self, algorithm: str) -> Dict[str, Any]:
        """Get default parameters for an algorithm."""
        algorithm_config = self._get_algorithm_config(algorithm)
        if not algorithm_config:
            return {}
        
        parameters_config = algorithm_config.get('parameters', {})
        defaults = {}
        
        for param_name, param_config in parameters_config.items():
            defaults[param_name] = param_config.get('default')
        
        return defaults
    
    def reset_parameters(self, algorithm: str, mode: str) -> None:
        """Reset parameters to defaults."""
        if f"validated_params_{algorithm}_{mode}" in st.session_state:
            del st.session_state[f"validated_params_{algorithm}_{mode}"]
        if f"validation_timestamp_{algorithm}_{mode}" in st.session_state:
            del st.session_state[f"validation_timestamp_{algorithm}_{mode}"]
        
        st.rerun()
