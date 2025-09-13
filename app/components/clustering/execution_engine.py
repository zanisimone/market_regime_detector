"""
Execution engine for clustering algorithms.

This module provides a unified interface for executing different clustering
algorithms with progress tracking and error handling.
"""

import streamlit as st
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional, Callable, Tuple, List
from .utils import ClusteringUtils
from .validation import ParameterValidator


class ClusteringExecutionEngine:
    """Unified execution engine for clustering algorithms."""
    
    def __init__(self):
        self.validator = ParameterValidator()
        self.utils = ClusteringUtils()
    
    def execute_algorithm(self, algorithm: str, mode: str, params: Dict[str, Any], 
                         output_dir: Optional[Path] = None, 
                         progress_callback: Optional[Callable] = None) -> Tuple[bool, Dict[str, Any]]:
        """
        Execute a clustering algorithm with the given parameters.
        
        Args:
            algorithm: Algorithm name
            mode: Training mode (split/rolling)
            params: Algorithm parameters
            output_dir: Output directory for results
            progress_callback: Optional callback for progress updates
            
        Returns:
            Tuple of (success, results_dict)
        """
        # Validate inputs
        if not self._validate_execution_inputs(algorithm, mode, params):
            return False, {}
        
        # Prepare output directory
        if output_dir is None:
            output_dir = self.utils.get_data_paths()["reports"] / f"{algorithm}_{mode}"
        
        self.utils.ensure_directory_exists(output_dir)
        
        # Create execution command
        try:
            cmd = self.utils.create_execution_command(algorithm, mode, params, output_dir)
        except Exception as e:
            st.error(f"Error creating execution command: {e}")
            return False, {}
        
        # Execute with progress tracking
        success, stdout, stderr = self._execute_with_progress(cmd, progress_callback)
        
        if success:
            # Load results
            results = self.utils.load_clustering_results(output_dir, algorithm)
            results["execution_info"] = {
                "algorithm": algorithm,
                "mode": mode,
                "parameters": params,
                "output_dir": str(output_dir),
                "timestamp": time.time(),
                "command": " ".join(cmd)
            }
            return True, results
        else:
            st.error(f"Execution failed: {stderr}")
            return False, {"error": stderr, "stdout": stdout}
    
    def _validate_execution_inputs(self, algorithm: str, mode: str, params: Dict[str, Any]) -> bool:
        """Validate inputs before execution."""
        # Check if algorithm is supported
        config = self.utils.load_algorithm_config(algorithm)
        if not config:
            st.error(f"Algorithm not supported: {algorithm}")
            return False
        
        # Check if mode is supported
        supported_modes = config.get('modes', [])
        if mode not in supported_modes:
            st.error(f"Mode {mode} not supported for algorithm {algorithm}")
            return False
        
        # Validate parameters
        is_valid, errors = self.validator.validate_algorithm_params(algorithm, params)
        if not is_valid:
            st.error("Parameter validation failed:")
            for error in errors:
                st.error(f"• {error}")
            return False
        
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
                    st.error("Date validation failed:")
                    for error in errors:
                        st.error(f"• {error}")
                    return False
        
        return True
    
    def _execute_with_progress(self, cmd: List[str], progress_callback: Optional[Callable] = None) -> Tuple[bool, str, str]:
        """Execute command with progress tracking."""
        if progress_callback:
            progress_callback(0, 4, "Starting execution...")
        
        try:
            # Start process
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            if progress_callback:
                progress_callback(1, 4, "Process started, waiting for completion...")
            
            # Wait for completion with timeout
            timeout = 300  # 5 minutes
            start_time = time.time()
            
            stdout_lines = []
            stderr_lines = []
            
            while True:
                # Check if process is still running
                return_code = process.poll()
                if return_code is not None:
                    # Process finished
                    break
                
                # Check timeout
                if time.time() - start_time > timeout:
                    process.terminate()
                    return False, "", f"Process timed out after {timeout} seconds"
                
                # Read output
                if process.stdout:
                    line = process.stdout.readline()
                    if line:
                        stdout_lines.append(line)
                        if progress_callback:
                            progress_callback(2, 4, f"Processing: {line.strip()}")
                
                time.sleep(0.1)  # Small delay to prevent busy waiting
            
            # Get final output
            stdout, stderr = process.communicate()
            stdout_lines.extend(stdout.splitlines())
            stderr_lines.extend(stderr.splitlines())
            
            if progress_callback:
                progress_callback(3, 4, "Process completed, loading results...")
            
            success = return_code == 0
            
            if progress_callback:
                progress_callback(4, 4, "Execution completed successfully!" if success else "Execution failed!")
            
            return success, "\n".join(stdout_lines), "\n".join(stderr_lines)
            
        except Exception as e:
            if progress_callback:
                progress_callback(4, 4, f"Execution error: {str(e)}")
            return False, "", str(e)
    
    def get_available_algorithms(self) -> List[str]:
        """Get list of available algorithms."""
        try:
            import json
            config_path = Path(__file__).parent.parent.parent / "configs" / "algorithms.json"
            with open(config_path, 'r') as f:
                algorithms_config = json.load(f)
            return list(algorithms_config.keys())
        except Exception:
            return []
    
    def get_algorithm_modes(self, algorithm: str) -> List[str]:
        """Get supported modes for an algorithm."""
        config = self.utils.load_algorithm_config(algorithm)
        if not config:
            return []
        return config.get('modes', [])
    
    def estimate_execution_time(self, algorithm: str, mode: str, params: Dict[str, Any]) -> int:
        """
        Estimate execution time in seconds.
        
        Args:
            algorithm: Algorithm name
            mode: Training mode
            params: Algorithm parameters
            
        Returns:
            Estimated time in seconds
        """
        # Base times for different algorithms (in seconds)
        base_times = {
            "kmeans": 30,
            "gmm": 60,
            "hmm": 120
        }
        
        # Mode multipliers
        mode_multipliers = {
            "split": 1.0,
            "rolling": 3.0
        }
        
        base_time = base_times.get(algorithm, 60)
        mode_multiplier = mode_multipliers.get(mode, 1.0)
        
        # Adjust for parameters
        if "n_clusters" in params:
            base_time *= (params["n_clusters"] / 3)  # Assume 3 is default
        
        if "n_init" in params:
            base_time *= (params["n_init"] / 10)  # Assume 10 is default
        
        return int(base_time * mode_multiplier)
    
    def cancel_execution(self, process_id: Optional[int] = None) -> bool:
        """Cancel a running execution."""
        # This would need to be implemented with proper process management
        # For now, return False as we don't have process tracking
        return False
