"""
Model management utilities for dynamic model selection and loading.

This module provides functions to discover, load, and manage different
clustering models and their outputs.
"""

import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from app.app_config import project_root


@dataclass
class ModelInfo:
    """Container for model information."""
    name: str
    algorithm: str
    mode: str
    labels_file: Path
    has_data: bool
    date_range: Optional[Tuple[str, str]] = None
    num_regimes: Optional[int] = None


class ModelManager:
    """Manages discovery and loading of clustering models."""
    
    def __init__(self):
        self.reports_dir = project_root() / "reports"
    
    def get_available_models(self) -> List[ModelInfo]:
        """
        Discover all available clustering models in the reports directory.
        
        Returns:
            List of ModelInfo objects for available models
        """
        models = []
        
        if not self.reports_dir.exists():
            return models
        
        # Look for model directories
        for model_dir in self.reports_dir.iterdir():
            if not model_dir.is_dir():
                continue
                
            # Parse model name (e.g., "kmeans_split" -> algorithm="kmeans", mode="split")
            parts = model_dir.name.split('_')
            if len(parts) != 2:
                continue
                
            algorithm, mode = parts
            
            # Look for labels file (CSV or Parquet)
            labels_file = None
            
            # Try different naming patterns
            patterns = [
                f"{model_dir.name}_labels.csv",  # kmeans_labels.csv
                f"{model_dir.name}_labels.parquet",  # gmm_labels.parquet, hmm_labels.parquet
                "rolling_labels.parquet",  # for rolling models
                "train_labels.parquet",  # fallback for train labels
                "test_labels.parquet",   # fallback for test labels
            ]
            
            for pattern in patterns:
                potential_file = model_dir / pattern
                if potential_file.exists():
                    labels_file = potential_file
                    break
            
            if labels_file is None:
                continue
            
            # Try to load and analyze the labels file
            has_data, date_range, num_regimes = self._analyze_labels_file(labels_file)
            
            model_info = ModelInfo(
                name=model_dir.name,
                algorithm=algorithm,
                mode=mode,
                labels_file=labels_file,
                has_data=has_data,
                date_range=date_range,
                num_regimes=num_regimes
            )
            
            models.append(model_info)
        
        # Sort by algorithm, then mode
        models.sort(key=lambda x: (x.algorithm, x.mode))
        return models
    
    def _analyze_labels_file(self, labels_file: Path) -> Tuple[bool, Optional[Tuple[str, str]], Optional[int]]:
        """
        Analyze a labels file to extract basic information.
        
        Args:
            labels_file: Path to the labels file
            
        Returns:
            Tuple of (has_data, date_range, num_regimes)
        """
        try:
            # Load the file
            if labels_file.suffix == '.csv':
                df = pd.read_csv(labels_file)
            else:  # .parquet
                df = pd.read_parquet(labels_file)
            
            if df.empty:
                return False, None, None
            
            # Check for required columns
            if 'regime' not in df.columns:
                return False, None, None
            
            # Handle date column - it might be in the index or in a column
            if 'date' in df.columns:
                # Date is in a column
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
                df = df.dropna(subset=['date'])
            elif isinstance(df.index, pd.DatetimeIndex):
                # Date is in the index
                df = df.reset_index()
                # The index column might be named 'index' or have a date-like name
                date_col = df.columns[0]  # First column after reset_index
                df = df.rename(columns={date_col: 'date'})
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
                df = df.dropna(subset=['date'])
            else:
                # Try to find a date-like column
                date_col = None
                for col in df.columns:
                    if col != 'regime':
                        try:
                            test_dates = pd.to_datetime(df[col], errors='coerce')
                            if test_dates.notna().mean() > 0.8:  # 80% valid dates
                                date_col = col
                                break
                        except:
                            continue
                
                if date_col is None:
                    return False, None, None
                
                df = df.rename(columns={date_col: 'date'})
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
                df = df.dropna(subset=['date'])
            
            if df.empty:
                return False, None, None
            
            # Extract information
            has_data = True
            date_range = (df['date'].min().strftime('%Y-%m-%d'), 
                         df['date'].max().strftime('%Y-%m-%d'))
            num_regimes = df['regime'].nunique()
            
            return has_data, date_range, num_regimes
            
        except Exception:
            return False, None, None
    
    def load_model_labels(self, model_name: str) -> Optional[pd.DataFrame]:
        """
        Load labels for a specific model.
        
        Args:
            model_name: Name of the model (e.g., "kmeans_split")
            
        Returns:
            DataFrame with date and regime columns, or None if not found
        """
        models = self.get_available_models()
        model_info = next((m for m in models if m.name == model_name), None)
        
        if model_info is None or not model_info.has_data:
            return None
        
        try:
            # Load the main labels file
            if model_info.labels_file.suffix == '.csv':
                df = pd.read_csv(model_info.labels_file)
            else:  # .parquet
                df = pd.read_parquet(model_info.labels_file)
            
            # Handle date column - it might be in the index or in a column
            if 'date' in df.columns:
                # Date is in a column
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
                df = df.dropna(subset=['date'])
            elif isinstance(df.index, pd.DatetimeIndex):
                # Date is in the index
                df = df.reset_index()
                # The index column might be named 'index' or have a date-like name
                date_col = df.columns[0]  # First column after reset_index
                df = df.rename(columns={date_col: 'date'})
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
                df = df.dropna(subset=['date'])
            else:
                # Try to find a date-like column
                date_col = None
                for col in df.columns:
                    if col != 'regime':
                        try:
                            test_dates = pd.to_datetime(df[col], errors='coerce')
                            if test_dates.notna().mean() > 0.8:  # 80% valid dates
                                date_col = col
                                break
                        except:
                            continue
                
                if date_col is None:
                    return None
                
                df = df.rename(columns={date_col: 'date'})
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
                df = df.dropna(subset=['date'])
            
            if df.empty:
                return None
            
            # For split models, always try to combine train and test data
            if model_info.mode == 'split':
                train_file = model_info.labels_file.parent / 'train_labels.parquet'
                test_file = model_info.labels_file.parent / 'test_labels.parquet'
                
                # If we loaded train, also load test
                if model_info.labels_file.name == 'train_labels.parquet' and test_file.exists():
                    try:
                        test_df = pd.read_parquet(test_file)
                        if isinstance(test_df.index, pd.DatetimeIndex):
                            test_df = test_df.reset_index()
                            test_df = test_df.rename(columns={'index': 'date'})
                        test_df['date'] = pd.to_datetime(test_df['date'], errors='coerce')
                        test_df = test_df.dropna(subset=['date'])
                        
                        # Combine train and test
                        combined_df = pd.concat([df, test_df], ignore_index=True)
                        combined_df = combined_df.sort_values('date').drop_duplicates(subset=['date'])
                        df = combined_df
                    except Exception:
                        pass  # Use only train data
                
                # If we loaded test, also load train
                elif model_info.labels_file.name == 'test_labels.parquet' and train_file.exists():
                    try:
                        train_df = pd.read_parquet(train_file)
                        if isinstance(train_df.index, pd.DatetimeIndex):
                            train_df = train_df.reset_index()
                            train_df = train_df.rename(columns={'index': 'date'})
                        train_df['date'] = pd.to_datetime(train_df['date'], errors='coerce')
                        train_df = train_df.dropna(subset=['date'])
                        
                        # Combine train and test
                        combined_df = pd.concat([train_df, df], ignore_index=True)
                        combined_df = combined_df.sort_values('date').drop_duplicates(subset=['date'])
                        df = combined_df
                    except Exception:
                        pass  # Use only test data
            
            # Return only date and regime columns
            return df[['date', 'regime']].sort_values('date')
            
        except Exception:
            return None
    
    def get_model_info(self, model_name: str) -> Optional[ModelInfo]:
        """
        Get detailed information about a specific model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            ModelInfo object or None if not found
        """
        models = self.get_available_models()
        return next((m for m in models if m.name == model_name), None)
    
    def get_model_display_name(self, model_name: str) -> str:
        """
        Get a human-readable display name for a model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Display name (e.g., "K-Means (Split)")
        """
        model_info = self.get_model_info(model_name)
        if model_info is None:
            return model_name
        
        algorithm_names = {
            'kmeans': 'K-Means',
            'gmm': 'Gaussian Mixture Model',
            'hmm': 'Hidden Markov Model'
        }
        
        mode_names = {
            'split': 'Split',
            'rolling': 'Rolling'
        }
        
        algorithm = algorithm_names.get(model_info.algorithm, model_info.algorithm.title())
        mode = mode_names.get(model_info.mode, model_info.mode.title())
        
        return f"{algorithm} ({mode})"


# Global instance
model_manager = ModelManager()


def get_available_models() -> List[ModelInfo]:
    """Get all available models."""
    return model_manager.get_available_models()


def load_model_labels(model_name: str) -> Optional[pd.DataFrame]:
    """Load labels for a specific model."""
    return model_manager.load_model_labels(model_name)


def get_model_info(model_name: str) -> Optional[ModelInfo]:
    """Get information about a specific model."""
    return model_manager.get_model_info(model_name)


def get_model_display_name(model_name: str) -> str:
    """Get display name for a model."""
    return model_manager.get_model_display_name(model_name)
