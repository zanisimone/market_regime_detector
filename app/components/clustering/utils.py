"""
Utility functions for clustering components.

This module provides common utility functions used across clustering components
for data processing, file management, and helper operations.
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import json
import subprocess
import sys
from datetime import datetime


class ClusteringUtils:
    """Utility class for clustering operations."""
    
    @staticmethod
    def load_algorithm_config(algorithm: str) -> Optional[Dict]:
        """Load configuration for a specific algorithm."""
        try:
            config_path = Path(__file__).parent.parent.parent / "configs" / "algorithms.json"
            with open(config_path, 'r') as f:
                algorithms_config = json.load(f)
            return algorithms_config.get(algorithm)
        except Exception as e:
            st.error(f"Error loading algorithm configuration: {e}")
            return None
    
    @staticmethod
    def get_project_root() -> Path:
        """Get the project root directory."""
        return Path(__file__).parent.parent.parent.parent
    
    @staticmethod
    def get_data_paths() -> Dict[str, Path]:
        """Get standard data paths."""
        root = ClusteringUtils.get_project_root()
        return {
            "features": root / "data" / "processed" / "features.parquet",
            "panel": root / "data" / "processed" / "market_panel.parquet",
            "reports": root / "reports",
            "cache": root / "cache"
        }
    
    @staticmethod
    def ensure_directory_exists(path: Path) -> bool:
        """Ensure a directory exists, create if necessary."""
        try:
            path.mkdir(parents=True, exist_ok=True)
            return True
        except Exception as e:
            st.error(f"Error creating directory {path}: {e}")
            return False
    
    @staticmethod
    def format_file_size(size_bytes: int) -> str:
        """Format file size in human readable format."""
        if size_bytes == 0:
            return "0 B"
        
        size_names = ["B", "KB", "MB", "GB", "TB"]
        i = 0
        while size_bytes >= 1024 and i < len(size_names) - 1:
            size_bytes /= 1024.0
            i += 1
        
        return f"{size_bytes:.1f} {size_names[i]}"
    
    @staticmethod
    def get_file_info(file_path: Path) -> Dict[str, Any]:
        """Get information about a file."""
        if not file_path.exists():
            return {"exists": False}
        
        stat = file_path.stat()
        return {
            "exists": True,
            "size": stat.st_size,
            "size_formatted": ClusteringUtils.format_file_size(stat.st_size),
            "modified": datetime.fromtimestamp(stat.st_mtime),
            "is_file": file_path.is_file(),
            "is_dir": file_path.is_dir()
        }
    
    @staticmethod
    def validate_data_file(file_path: Path, expected_columns: Optional[List[str]] = None) -> Tuple[bool, str]:
        """
        Validate a data file for clustering.
        
        Args:
            file_path: Path to the data file
            expected_columns: List of expected column names
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not file_path.exists():
            return False, f"File does not exist: {file_path}"
        
        try:
            # Try to read the file
            if file_path.suffix == '.parquet':
                df = pd.read_parquet(file_path)
            elif file_path.suffix == '.csv':
                df = pd.read_csv(file_path)
            else:
                return False, f"Unsupported file format: {file_path.suffix}"
            
            # Check if file is empty
            if len(df) == 0:
                return False, "File is empty"
            
            # Check for required columns
            if expected_columns:
                missing_columns = set(expected_columns) - set(df.columns)
                if missing_columns:
                    return False, f"Missing required columns: {list(missing_columns)}"
            
            # Check for too many NaN values
            nan_ratio = df.isnull().sum().sum() / (len(df) * len(df.columns))
            if nan_ratio > 0.5:
                return False, f"Too many missing values: {nan_ratio:.1%}"
            
            return True, ""
            
        except Exception as e:
            return False, f"Error reading file: {str(e)}"
    
    @staticmethod
    def create_execution_command(algorithm: str, mode: str, params: Dict[str, Any], 
                               output_dir: Optional[Path] = None) -> List[str]:
        """
        Create command for executing clustering algorithm using existing scripts.
        
        Args:
            algorithm: Algorithm name
            mode: Training mode
            params: Algorithm parameters
            output_dir: Output directory
            
        Returns:
            Command as list of strings
        """
        # Map algorithm names to script names
        script_mapping = {
            "kmeans": {
                "split": "scripts.run_kmeans_split",
                "rolling": "scripts.run_kmeans_rolling"
            },
            "gmm": {
                "split": "scripts.run_gmm_split", 
                "rolling": "scripts.run_gmm_rolling"
            },
            "hmm": {
                "split": "scripts.run_hmm_split",
                "rolling": "scripts.run_hmm_rolling"
            }
        }
        
        if algorithm not in script_mapping:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        if mode not in script_mapping[algorithm]:
            raise ValueError(f"Mode {mode} not supported for {algorithm}")
        
        script_name = script_mapping[algorithm][mode]
        cmd = [sys.executable, "-m", script_name]
        
        # Add parameters based on mode
        if mode == "split":
            # Split mode parameters
            required_dates = ["train_start", "train_end", "test_start", "test_end"]
            for date_param in required_dates:
                if date_param in params:
                    cmd.extend([f"--{date_param.replace('_', '-')}", str(params[date_param])])
                else:
                    raise ValueError(f"Missing required parameter for split mode: {date_param}")
            
            # Algorithm parameters
            if "n_clusters" in params:
                cmd.extend(["--k", str(params["n_clusters"])])
            elif "n_components" in params:
                cmd.extend(["--k", str(params["n_components"])])
            elif "n_states" in params:
                cmd.extend(["--k", str(params["n_states"])])
            
            if "random_state" in params:
                cmd.extend(["--random-state", str(params["random_state"])])
            
            if "n_init" in params:
                cmd.extend(["--n-init", str(params["n_init"])])
            
            if "n_iter" in params:
                cmd.extend(["--n-iter", str(params["n_iter"])])
            
            if "covariance_type" in params:
                cmd.extend(["--covariance-type", str(params["covariance_type"])])
            
            # K-Means doesn't have max_iter parameter, skip it
            # if "max_iter" in params:
            #     cmd.extend(["--max-iter", str(params["max_iter"])])
        
        elif mode == "rolling":
            # Rolling mode parameters
            if "start" in params:
                cmd.extend(["--start", str(params["start"])])
            if "end" in params:
                cmd.extend(["--end", str(params["end"])])
            
            # Rolling window parameters
            if "window_size" in params:
                cmd.extend(["--lookback-days", str(params["window_size"])])
            elif "lookback_days" in params:
                cmd.extend(["--lookback-days", str(params["lookback_days"])])
            else:
                cmd.extend(["--lookback-days", "504"])  # Default
            
            if "step_size" in params:
                cmd.extend(["--step-days", str(params["step_size"])])
            elif "step_days" in params:
                cmd.extend(["--step-days", str(params["step_days"])])
            else:
                cmd.extend(["--step-days", "21"])  # Default
            
            if "oos_days" in params:
                cmd.extend(["--oos-days", str(params["oos_days"])])
            else:
                cmd.extend(["--oos-days", "21"])  # Default
            
            # Algorithm parameters
            if "n_clusters" in params:
                cmd.extend(["--k", str(params["n_clusters"])])
            elif "n_components" in params:
                cmd.extend(["--k", str(params["n_components"])])
            elif "n_states" in params:
                cmd.extend(["--k", str(params["n_states"])])
            
            if "random_state" in params:
                cmd.extend(["--random-state", str(params["random_state"])])
            
            if "n_init" in params:
                cmd.extend(["--n-init", str(params["n_init"])])
            
            if "n_iter" in params:
                cmd.extend(["--n-iter", str(params["n_iter"])])
            
            if "covariance_type" in params:
                cmd.extend(["--covariance-type", str(params["covariance_type"])])
        
        # Add output directory
        if output_dir:
            cmd.extend(["--out-dir", str(output_dir)])
        
        return cmd
    
    @staticmethod
    def execute_command(cmd: List[str], timeout: int = 300) -> Tuple[bool, str, str]:
        """
        Execute a command and return results.
        
        Args:
            cmd: Command to execute
            timeout: Timeout in seconds
            
        Returns:
            Tuple of (success, stdout, stderr)
        """
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False
            )
            
            return result.returncode == 0, result.stdout, result.stderr
            
        except subprocess.TimeoutExpired:
            return False, "", f"Command timed out after {timeout} seconds"
        except Exception as e:
            return False, "", f"Error executing command: {str(e)}"
    
    @staticmethod
    def load_clustering_results(output_dir: Path, algorithm: str) -> Dict[str, Any]:
        """
        Load clustering results from output directory.
        
        Args:
            output_dir: Directory containing results
            algorithm: Algorithm name
            
        Returns:
            Dictionary containing loaded results
        """
        results = {}
        
        if not output_dir.exists():
            return results
        
        # Define expected files for each algorithm
        expected_files = {
            "kmeans": {
                "labels": ["train_labels.parquet", "test_labels.parquet", "kmeans_labels.csv"],
                "centers": ["kmeans_centers.csv"],
                "model": ["kmeans_model.pkl"]
            },
            "gmm": {
                "labels": ["train_labels.parquet", "test_labels.parquet"],
                "centers": ["gmm_means.csv"],
                "model": ["gmm_model.pkl"]
            },
            "hmm": {
                "labels": ["train_labels.parquet", "test_labels.parquet"],
                "centers": ["hmm_means.csv"],
                "model": ["hmm_model.pkl"]
            }
        }
        
        if algorithm not in expected_files:
            return results
        
        # Load labels (priority: rolling files first, then split files)
        labels_loaded = False
        
        # First check for rolling mode files (rolling_labels.parquet)
        # For rolling mode, check reports directory only (no data/processed)
        reports_dir = output_dir.parent.parent / "reports" / output_dir.name
        rolling_labels_path = reports_dir / "rolling_labels.parquet"
        
        if rolling_labels_path.exists():
            try:
                df = pd.read_parquet(rolling_labels_path)
                results["labels"] = df
                # Also try to load rolling schedule
                rolling_schedule_path = reports_dir / "rolling_schedule.csv"
                if rolling_schedule_path.exists():
                    results["rolling_schedule"] = pd.read_csv(rolling_schedule_path)
                labels_loaded = True
            except Exception as e:
                st.warning(f"Error loading rolling results: {e}")
        
        # If not rolling mode, try to load split mode files
        if not labels_loaded:
            # First try to load parquet files from model directory
            label_files = expected_files[algorithm]["labels"]
            if isinstance(label_files, str):
                label_files = [label_files]
            
            for label_file in label_files:
                file_path = output_dir / label_file
                if file_path.exists():
                    try:
                        if file_path.suffix == '.parquet':
                            df = pd.read_parquet(file_path)
                            results["labels"] = df
                            labels_loaded = True
                            break
                    except Exception as e:
                        st.warning(f"Error loading labels from {label_file}: {e}")
            
            # If not found in model directory, try CSV from reports directory
            if not labels_loaded:
                reports_dir = output_dir.parent.parent / "reports" / output_dir.name
                if reports_dir.exists():
                    for label_file in label_files:
                        file_path = reports_dir / label_file
                        if file_path.exists():
                            try:
                                if file_path.suffix == '.csv':
                                    df = pd.read_csv(file_path, parse_dates=['date'] if 'date' in pd.read_csv(file_path, nrows=1).columns else None)
                                    if 'date' in df.columns:
                                        df = df.set_index('date')
                                    results["labels"] = df
                                    labels_loaded = True
                                    break
                            except Exception as e:
                                st.warning(f"Error loading labels from {label_file}: {e}")
        
        if not labels_loaded:
            st.warning(f"No labels found for {algorithm}")
        
        # Load centers/means
        for center_file in expected_files[algorithm]["centers"]:
            file_path = output_dir / center_file
            if file_path.exists():
                try:
                    df = pd.read_csv(file_path)
                    results["centers"] = df
                    break
                except Exception as e:
                    st.warning(f"Error loading centers from {center_file}: {e}")
        
        # Load model
        for model_file in expected_files[algorithm]["model"]:
            file_path = output_dir / model_file
            if file_path.exists():
                try:
                    import joblib
                    results["model"] = joblib.load(file_path)
                    break
                except Exception as e:
                    st.warning(f"Error loading model from {model_file}: {e}")
        
        return results
    
    @staticmethod
    def calculate_regime_statistics(labels: pd.Series) -> Dict[str, Any]:
        """
        Calculate basic statistics for regime labels.
        
        Args:
            labels: Series of regime labels
            
        Returns:
            Dictionary of statistics
        """
        if labels.empty:
            return {}
        
        stats = {
            "total_observations": len(labels),
            "unique_regimes": labels.nunique(),
            "regime_counts": labels.value_counts().to_dict(),
            "regime_proportions": (labels.value_counts(normalize=True) * 100).to_dict(),
            "most_common_regime": labels.mode().iloc[0] if not labels.empty else None,
            "regime_diversity": labels.nunique() / len(labels) if len(labels) > 0 else 0
        }
        
        return stats
    
    @staticmethod
    def format_timestamp(timestamp: datetime) -> str:
        """Format timestamp for display."""
        return timestamp.strftime("%Y-%m-%d %H:%M:%S")
    
    @staticmethod
    def create_progress_callback(progress_bar, status_text):
        """Create a progress callback for long-running operations."""
        def callback(step: int, total_steps: int, message: str = ""):
            if progress_bar:
                progress_bar.progress(step / total_steps)
            if status_text:
                status_text.text(f"Step {step}/{total_steps}: {message}")
        
        return callback
    
    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """Sanitize filename for safe file operations."""
        import re
        # Remove or replace invalid characters
        filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
        # Remove leading/trailing spaces and dots
        filename = filename.strip(' .')
        # Ensure it's not empty
        if not filename:
            filename = "untitled"
        return filename
    
    @staticmethod
    def get_algorithm_display_name(algorithm: str) -> str:
        """Get display name for algorithm."""
        config = ClusteringUtils.load_algorithm_config(algorithm)
        return config.get('name', algorithm.title()) if config else algorithm.title()
    
    @staticmethod
    def create_download_link(data: Any, filename: str, file_type: str = "csv") -> str:
        """Create a download link for data."""
        import base64
        import io
        
        if file_type == "csv":
            if isinstance(data, pd.DataFrame):
                csv = data.to_csv(index=True)
            else:
                csv = str(data)
            b64 = base64.b64encode(csv.encode()).decode()
            return f'<a href="data:file/csv;base64,{b64}" download="{filename}.csv">Download {filename}</a>'
        
        elif file_type == "json":
            json_str = json.dumps(data, indent=2, default=str)
            b64 = base64.b64encode(json_str.encode()).decode()
            return f'<a href="data:file/json;base64,{b64}" download="{filename}.json">Download {filename}</a>'
        
        return ""
