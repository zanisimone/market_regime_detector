"""
Parameter validation system for clustering algorithms.

This module provides comprehensive validation for algorithm parameters,
date ranges, and other inputs to ensure robust execution.
"""

import pandas as pd
from datetime import datetime, date
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import streamlit as st


class ParameterValidator:
    """Comprehensive parameter validation for clustering algorithms."""
    
    def __init__(self):
        self.error_messages = {
            "INVALID_DATE_FORMAT": "Invalid date format. Please use YYYY-MM-DD",
            "INVALID_DATE_RANGE": "Invalid date range. Start date must be before end date",
            "INVALID_TRAIN_TEST_SPLIT": "Training period must end before test period starts",
            "INVALID_PARAMETER_VALUE": "Parameter value out of allowed range",
            "MISSING_REQUIRED_PARAMETER": "Required parameter is missing",
            "INVALID_FILE_PATH": "File path does not exist or is not accessible",
            "INVALID_ALGORITHM": "Selected algorithm is not supported",
            "INVALID_MODE": "Selected training mode is not supported for this algorithm"
        }
    
    def validate_date_input(self, date_str: str, field_name: str) -> Tuple[bool, Optional[datetime], str]:
        """
        Validate a single date input string.
        
        Args:
            date_str: Date string to validate
            field_name: Name of the field for error messages
            
        Returns:
            Tuple of (is_valid, parsed_date, error_message)
        """
        if not date_str or not date_str.strip():
            return False, None, f"{field_name} cannot be empty"
        
        try:
            parsed_date = pd.to_datetime(date_str).to_pydatetime()
            return True, parsed_date, ""
        except (ValueError, TypeError) as e:
            return False, None, f"{field_name}: {self.error_messages['INVALID_DATE_FORMAT']}"
    
    def validate_date_ranges(self, train_start: str, train_end: str, 
                           test_start: str, test_end: str) -> Tuple[bool, List[str]]:
        """
        Validate date ranges for split mode training.
        
        Args:
            train_start: Training start date
            train_end: Training end date  
            test_start: Test start date
            test_end: Test end date
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Validate individual dates
        date_fields = [
            ("Training Start", train_start),
            ("Training End", train_end),
            ("Test Start", test_start),
            ("Test End", test_end)
        ]
        
        parsed_dates = {}
        for field_name, date_str in date_fields:
            is_valid, parsed_date, error_msg = self.validate_date_input(date_str, field_name)
            if not is_valid:
                errors.append(error_msg)
            else:
                parsed_dates[field_name.lower().replace(" ", "_")] = parsed_date
        
        if errors:
            return False, errors
        
        # Validate date logic
        train_start_dt = parsed_dates["training_start"]
        train_end_dt = parsed_dates["training_end"]
        test_start_dt = parsed_dates["test_start"]
        test_end_dt = parsed_dates["test_end"]
        
        # Check if start < end for each period
        if train_start_dt >= train_end_dt:
            errors.append("Training start must be before training end")
        
        if test_start_dt >= test_end_dt:
            errors.append("Test start must be before test end")
        
        # Check if training ends before test starts
        if train_end_dt >= test_start_dt:
            errors.append("Training period must end before test period starts")
        
        # Check if dates are not too far in the future
        today = datetime.now()
        if test_end_dt > today:
            errors.append("Test end date cannot be in the future")
        
        # Check if dates are not too far in the past (before 2000)
        min_date = datetime(2000, 1, 1)
        if train_start_dt < min_date:
            errors.append("Training start date cannot be before 2000-01-01")
        if test_start_dt < min_date:
            errors.append("Test start date cannot be before 2000-01-01")
        
        return len(errors) == 0, errors
    
    def validate_rolling_params(self, window_size: int, step_size: int, 
                              min_periods: int = 252) -> Tuple[bool, List[str]]:
        """
        Validate rolling window parameters.
        
        Args:
            window_size: Size of rolling window
            step_size: Step size for rolling
            min_periods: Minimum periods required
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        if window_size < min_periods:
            errors.append(f"Window size ({window_size}) must be at least {min_periods}")
        
        if step_size < 1:
            errors.append("Step size must be at least 1")
        
        if step_size > window_size:
            errors.append("Step size cannot be larger than window size")
        
        return len(errors) == 0, errors
    
    def validate_algorithm_params(self, algorithm: str, params: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate algorithm-specific parameters.
        
        Args:
            algorithm: Algorithm name
            params: Parameter dictionary
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Load algorithm configuration
        try:
            import json
            config_path = Path(__file__).parent.parent.parent / "configs" / "algorithms.json"
            with open(config_path, 'r') as f:
                algorithms_config = json.load(f)
            
            if algorithm not in algorithms_config:
                errors.append(f"Unknown algorithm: {algorithm}")
                return False, errors
            
            algo_config = algorithms_config[algorithm]
            param_configs = algo_config.get("parameters", {})
            
            # Validate each parameter
            for param_name, param_value in params.items():
                if param_name not in param_configs:
                    continue  # Skip unknown parameters
                
                param_config = param_configs[param_name]
                param_type = param_config.get("type")
                param_min = param_config.get("min")
                param_max = param_config.get("max")
                
                if param_type == "int":
                    if not isinstance(param_value, int):
                        errors.append(f"{param_name} must be an integer")
                    elif param_min is not None and param_value < param_min:
                        errors.append(f"{param_name} must be at least {param_min}")
                    elif param_max is not None and param_value > param_max:
                        errors.append(f"{param_name} must be at most {param_max}")
                
                elif param_type == "select":
                    valid_options = param_config.get("options", [])
                    if param_value not in valid_options:
                        errors.append(f"{param_name} must be one of: {', '.join(valid_options)}")
            
        except Exception as e:
            errors.append(f"Error loading algorithm configuration: {str(e)}")
        
        return len(errors) == 0, errors
    
    def validate_file_paths(self, file_paths: Dict[str, Path]) -> Tuple[bool, List[str]]:
        """
        Validate that required file paths exist.
        
        Args:
            file_paths: Dictionary of file path names to Path objects
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        for name, path in file_paths.items():
            if not path.exists():
                errors.append(f"{name} file not found: {path}")
            elif not path.is_file():
                errors.append(f"{name} path is not a file: {path}")
        
        return len(errors) == 0, errors
    
    def validate_algorithm_mode_combination(self, algorithm: str, mode: str) -> Tuple[bool, str]:
        """
        Validate that the algorithm supports the selected mode.
        
        Args:
            algorithm: Algorithm name
            mode: Training mode (split/rolling)
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            import json
            config_path = Path(__file__).parent.parent.parent / "configs" / "algorithms.json"
            with open(config_path, 'r') as f:
                algorithms_config = json.load(f)
            
            if algorithm not in algorithms_config:
                return False, f"Unknown algorithm: {algorithm}"
            
            supported_modes = algorithms_config[algorithm].get("modes", [])
            if mode not in supported_modes:
                return False, f"Algorithm {algorithm} does not support mode {mode}"
            
            return True, ""
            
        except Exception as e:
            return False, f"Error validating algorithm-mode combination: {str(e)}"
    
    def display_validation_errors(self, errors: List[str]) -> None:
        """
        Display validation errors in Streamlit.
        
        Args:
            errors: List of error messages
        """
        if errors:
            st.error("Validation Errors:")
            for error in errors:
                st.error(f"• {error}")
    
    def display_validation_success(self, message: str = "All parameters are valid") -> None:
        """
        Display validation success message in Streamlit.
        
        Args:
            message: Success message to display
        """
        st.success(message)


class ValidationResult:
    """Container for validation results."""
    
    def __init__(self, is_valid: bool, errors: List[str] = None, warnings: List[str] = None):
        self.is_valid = is_valid
        self.errors = errors or []
        self.warnings = warnings or []
    
    def add_error(self, error: str):
        """Add an error message."""
        self.errors.append(error)
        self.is_valid = False
    
    def add_warning(self, warning: str):
        """Add a warning message."""
        self.warnings.append(warning)
    
    def has_errors(self) -> bool:
        """Check if there are any errors."""
        return len(self.errors) > 0
    
    def has_warnings(self) -> bool:
        """Check if there are any warnings."""
        return len(self.warnings) > 0
    
    def display_in_streamlit(self):
        """Display validation results in Streamlit."""
        if self.has_errors():
            st.error("Validation Errors:")
            for error in self.errors:
                st.error(f"• {error}")
        
        if self.has_warnings():
            st.warning("Validation Warnings:")
            for warning in self.warnings:
                st.warning(f"• {warning}")
        
        if self.is_valid and not self.has_warnings():
            st.success("✅ All parameters are valid")
