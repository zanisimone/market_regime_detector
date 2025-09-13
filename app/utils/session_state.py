"""
Session state management utilities for the Streamlit app.

This module provides functions to manage global state across pages,
including model selection and navigation state.
"""

import streamlit as st
from typing import Optional, List, Dict, Any
from app.utils.model_manager import ModelInfo, get_available_models


# Session state keys
class SessionKeys:
    """Constants for session state keys."""
    SELECTED_MODEL = "selected_model"
    AVAILABLE_MODELS = "available_models"
    LAST_RUN_MODEL = "last_run_model"
    MODEL_SELECTION_SOURCE = "model_selection_source"  # "manual" or "from_page2"
    NAVIGATION_HISTORY = "navigation_history"


def initialize_session_state():
    """Initialize session state with default values."""
    if SessionKeys.SELECTED_MODEL not in st.session_state:
        st.session_state[SessionKeys.SELECTED_MODEL] = None
    
    if SessionKeys.AVAILABLE_MODELS not in st.session_state:
        st.session_state[SessionKeys.AVAILABLE_MODELS] = []
    
    if SessionKeys.LAST_RUN_MODEL not in st.session_state:
        st.session_state[SessionKeys.LAST_RUN_MODEL] = None
    
    if SessionKeys.MODEL_SELECTION_SOURCE not in st.session_state:
        st.session_state[SessionKeys.MODEL_SELECTION_SOURCE] = "manual"
    
    if SessionKeys.NAVIGATION_HISTORY not in st.session_state:
        st.session_state[SessionKeys.NAVIGATION_HISTORY] = []


def refresh_available_models():
    """Refresh the list of available models from disk."""
    models = get_available_models()
    st.session_state[SessionKeys.AVAILABLE_MODELS] = models
    return models


def get_available_models_from_session() -> List[ModelInfo]:
    """Get available models from session state, refreshing if needed."""
    if not st.session_state[SessionKeys.AVAILABLE_MODELS]:
        return refresh_available_models()
    return st.session_state[SessionKeys.AVAILABLE_MODELS]


def set_selected_model(model_name: str, source: str = "manual"):
    """
    Set the currently selected model.
    
    Args:
        model_name: Name of the model to select
        source: Source of selection ("manual" or "from_page2")
    """
    st.session_state[SessionKeys.SELECTED_MODEL] = model_name
    st.session_state[SessionKeys.MODEL_SELECTION_SOURCE] = source


def get_selected_model() -> Optional[str]:
    """Get the currently selected model."""
    return st.session_state.get(SessionKeys.SELECTED_MODEL)


def set_last_run_model(model_name: str):
    """Set the last model that was run from page 2."""
    st.session_state[SessionKeys.LAST_RUN_MODEL] = model_name


def get_last_run_model() -> Optional[str]:
    """Get the last model that was run from page 2."""
    return st.session_state.get(SessionKeys.LAST_RUN_MODEL)


def get_model_selection_source() -> str:
    """Get the source of the current model selection."""
    return st.session_state.get(SessionKeys.MODEL_SELECTION_SOURCE, "manual")


def add_to_navigation_history(page: str, context: Optional[Dict[str, Any]] = None):
    """Add a page to navigation history."""
    if SessionKeys.NAVIGATION_HISTORY not in st.session_state:
        st.session_state[SessionKeys.NAVIGATION_HISTORY] = []
    
    entry = {
        "page": page,
        "context": context or {},
        "timestamp": st.session_state.get("_last_rerun_time", 0)
    }
    
    st.session_state[SessionKeys.NAVIGATION_HISTORY].append(entry)
    
    # Keep only last 10 entries
    if len(st.session_state[SessionKeys.NAVIGATION_HISTORY]) > 10:
        st.session_state[SessionKeys.NAVIGATION_HISTORY] = st.session_state[SessionKeys.NAVIGATION_HISTORY][-10:]


def get_navigation_history() -> List[Dict[str, Any]]:
    """Get navigation history."""
    return st.session_state.get(SessionKeys.NAVIGATION_HISTORY, [])


def clear_model_selection():
    """Clear the current model selection."""
    st.session_state[SessionKeys.SELECTED_MODEL] = None
    st.session_state[SessionKeys.MODEL_SELECTION_SOURCE] = "manual"


def get_default_model_for_page3() -> Optional[str]:
    """
    Get the default model to use for page 3.
    
    Priority:
    1. Last run model (if available and has data)
    2. First available model with data
    3. None (no models available)
    """
    available_models = get_available_models_from_session()
    
    # Filter models that have data
    valid_models = [m for m in available_models if m.has_data]
    
    if not valid_models:
        return None
    
    # Check if last run model is available and valid
    last_run = get_last_run_model()
    if last_run:
        last_run_model = next((m for m in valid_models if m.name == last_run), None)
        if last_run_model:
            return last_run_model.name
    
    # Fall back to first available model
    return valid_models[0].name


def should_show_model_selector() -> bool:
    """
    Determine if the model selector should be shown on page 3.
    
    Returns True if:
    - No model is selected, OR
    - Model selection source is manual (not from page 2), OR
    - Multiple models are available
    """
    selected_model = get_selected_model()
    available_models = get_available_models_from_session()
    valid_models = [m for m in available_models if m.has_data]
    
    # Always show if no model selected
    if not selected_model:
        return True
    
    # Always show if multiple models available
    if len(valid_models) > 1:
        return True
    
    # Show if selection source is manual
    if get_model_selection_source() == "manual":
        return True
    
    return False


def get_model_selector_options() -> List[tuple]:
    """
    Get options for the model selector dropdown.
    
    Returns:
        List of (display_name, model_name) tuples
    """
    from .model_manager import get_model_display_name
    
    available_models = get_available_models_from_session()
    valid_models = [m for m in available_models if m.has_data]
    
    options = []
    for model in valid_models:
        display_name = get_model_display_name(model.name)
        # Add additional info if available
        if model.date_range and model.num_regimes:
            display_name += f" ({model.date_range[0]} to {model.date_range[1]}, {model.num_regimes} regimes)"
        
        options.append((display_name, model.name))
    
    return options
