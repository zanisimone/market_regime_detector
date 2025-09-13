"""
Model selector component for choosing between available clustering models.

This component provides a unified interface for model selection across
different pages of the application.
"""

import streamlit as st
from typing import Optional, List, Dict, Any, Callable
from app.utils.session_state import (
    initialize_session_state,
    get_available_models_from_session,
    get_selected_model,
    set_selected_model,
    get_default_model_for_page3,
    should_show_model_selector,
    get_model_selector_options
)
from app.utils.model_manager import get_model_info
from app.utils.model_manager import ModelInfo


class ModelSelector:
    """Component for selecting clustering models."""
    
    def __init__(self, show_info: bool = True, show_validation: bool = True):
        """
        Initialize the model selector.
        
        Args:
            show_info: Whether to show model information
            show_validation: Whether to show validation messages
        """
        self.show_info = show_info
        self.show_validation = show_validation
        initialize_session_state()
    
    def render_selector(self, 
                       key: str = "model_selector",
                       default_model: Optional[str] = None,
                       on_change: Optional[Callable] = None) -> Optional[str]:
        """
        Render the model selector component.
        
        Args:
            key: Unique key for the component
            default_model: Default model to select
            on_change: Callback function when selection changes
            
        Returns:
            Selected model name or None
        """
        # Get available models
        available_models = get_available_models_from_session()
        valid_models = [m for m in available_models if m.has_data]
        
        if not valid_models:
            if self.show_validation:
                st.warning("⚠️ No clustering models available. Please run a model from the Clustering page first.")
            return None
        
        # Get current selection
        current_selection = get_selected_model()
        
        # Determine default selection
        if default_model and any(m.name == default_model for m in valid_models):
            default_selection = default_model
        elif current_selection and any(m.name == current_selection for m in valid_models):
            default_selection = current_selection
        else:
            default_selection = get_default_model_for_page3()
        
        # Get selector options
        options = get_model_selector_options()
        
        if not options:
            if self.show_validation:
                st.error("❌ No valid models found.")
            return None
        
        # Create selection mapping
        display_to_model = {display: model for display, model in options}
        
        # Find default index
        default_index = 0
        if default_selection:
            for i, (display, model) in enumerate(options):
                if model == default_selection:
                    default_index = i
                    break
        
        # Render selector
        selected_display = st.selectbox(
            "Select Model:",
            options=[opt[0] for opt in options],
            index=default_index,
            key=f"{key}_selectbox",
            help="Choose a clustering model to visualize"
        )
        
        selected_model = display_to_model[selected_display]
        
        # Update session state if selection changed
        if selected_model != current_selection:
            set_selected_model(selected_model, "manual")
            if on_change:
                on_change(selected_model)
        
        # Show model information
        if self.show_info and selected_model:
            self._render_model_info(selected_model)
        
        return selected_model
    
    def _render_model_info(self, model_name: str):
        """Render detailed information about the selected model."""
        model_info = get_model_info(model_name)
        if not model_info:
            return
        
        with st.expander("ℹ️ Model Information", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Algorithm", model_info.algorithm.upper())
                st.metric("Mode", model_info.mode.title())
                if model_info.num_regimes:
                    st.metric("Number of Regimes", model_info.num_regimes)
            
            with col2:
                if model_info.date_range:
                    st.metric("Start Date", model_info.date_range[0])
                    st.metric("End Date", model_info.date_range[1])
                
                st.metric("Data Status", "✅ Available" if model_info.has_data else "❌ Not Available")
            
            # Show file path
            st.caption(f"Labels file: `{model_info.labels_file.name}`")
    
    def render_compact_selector(self, 
                               key: str = "compact_model_selector",
                               default_model: Optional[str] = None) -> Optional[str]:
        """
        Render a compact model selector (just the dropdown).
        
        Args:
            key: Unique key for the component
            default_model: Default model to select
            
        Returns:
            Selected model name or None
        """
        # Temporarily disable info display
        original_show_info = self.show_info
        self.show_info = False
        
        try:
            return self.render_selector(key, default_model)
        finally:
            self.show_info = original_show_info
    
    def render_model_status(self, model_name: str):
        """
        Render status information for a specific model.
        
        Args:
            model_name: Name of the model to show status for
        """
        model_info = get_model_info(model_name)
        if not model_info:
            st.error(f"❌ Model '{model_name}' not found")
            return
        
        # Status indicator
        if model_info.has_data:
            st.success(f"✅ {model_name} - Data available")
        else:
            st.error(f"❌ {model_name} - No data available")
        
        # Quick info
        if model_info.date_range and model_info.num_regimes:
            st.caption(f"Period: {model_info.date_range[0]} to {model_info.date_range[1]} | Regimes: {model_info.num_regimes}")


def render_model_selector(show_info: bool = True, 
                         key: str = "model_selector",
                         default_model: Optional[str] = None) -> Optional[str]:
    """
    Convenience function to render a model selector.
    
    Args:
        show_info: Whether to show model information
        key: Unique key for the component
        default_model: Default model to select
        
    Returns:
        Selected model name or None
    """
    selector = ModelSelector(show_info=show_info)
    return selector.render_selector(key=key, default_model=default_model)


def render_compact_model_selector(key: str = "compact_model_selector",
                                 default_model: Optional[str] = None) -> Optional[str]:
    """
    Convenience function to render a compact model selector.
    
    Args:
        key: Unique key for the component
        default_model: Default model to select
        
    Returns:
        Selected model name or None
    """
    selector = ModelSelector(show_info=False)
    return selector.render_compact_selector(key=key, default_model=default_model)
