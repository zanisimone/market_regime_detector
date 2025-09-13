import streamlit as st
import pandas as pd
from pathlib import Path
import sys

# --- sys.path bootstrap (page) ---
ROOT = Path(__file__).resolve().parents[2]  # .../market_regime_detector
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# ---------------------------------

from app.app_config import project_root
from app.utils import initialize_session_state, set_selected_model
from app.components.regime_info import load_regime_catalog, show_regime_info
from app.components.clustering import (
    AlgorithmSelector, 
    ParameterForm, 
    ClusteringExecutionEngine,
    ResultsDisplay,
    ModelComparison
)


def main():
    """Enhanced clustering page with full algorithm support."""
    st.title("🔍 Clustering")

    # Initialize session state
    initialize_session_state()

    # Initialize components
    algorithm_selector = AlgorithmSelector()
    parameter_form = ParameterForm()
    execution_engine = ClusteringExecutionEngine()
    results_display = ResultsDisplay()
    model_comparison = ModelComparison()
    
    # Sidebar for algorithm selection
    with st.sidebar:
        st.header("🔧 Algorithm Configuration")
        
        # Algorithm and mode selection
        selected_algorithm, selected_mode = algorithm_selector.render_algorithm_selection()
        
        # Algorithm comparison
        if st.button("📊 Compare Algorithms"):
            algorithm_selector.render_algorithm_comparison()
    
    # Main content area
    if selected_algorithm and selected_mode:
        # Parameter configuration
        st.header("⚙️ Parameter Configuration")
        
        # Load current parameters from session state
        current_params = st.session_state.get(f"validated_params_{selected_algorithm}_{selected_mode}", {})
        
        # Render parameter form
        params = parameter_form.render_parameter_form(
            selected_algorithm, 
            selected_mode, 
            current_params
        )
        
        # Display parameter summary
        if params:
            parameter_form.render_parameter_summary(selected_algorithm, selected_mode, params)
        
        # Execution section
        st.header("🚀 Execution")
        
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            if st.button("▶️ Run Algorithm", type="primary"):
                # Validate parameters before execution
                if not params:
                    st.error("Please configure parameters first")
                else:
                    # Create progress tracking
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Create progress callback
                    def progress_callback(step, total, message):
                        progress_bar.progress(step / total)
                        status_text.text(f"Step {step}/{total}: {message}")
                    
                    # Execute algorithm
                    with st.spinner("Executing algorithm..."):
                        success, results = execution_engine.execute_algorithm(
                            selected_algorithm,
                            selected_mode,
                            params,
                            progress_callback=progress_callback
                        )
                    
                    if success:
                        st.session_state[f"results_{selected_algorithm}_{selected_mode}"] = results
                        st.success("✅ Algorithm executed successfully!")
                        
                        # Save the just-run model to session state for page 3
                        model_name = f"{selected_algorithm}_{selected_mode}"
                        set_selected_model(model_name)
                        st.session_state["last_run_model"] = model_name
                    else:
                        st.error("❌ Algorithm execution failed")
        
        with col2:
            if st.button("🔄 Reset Parameters"):
                parameter_form.reset_parameters(selected_algorithm, selected_mode)
        
        with col3:
            # Load features for comparison
            features_path = project_root() / "data" / "processed" / "features.parquet"
            if features_path.exists():
                features = pd.read_parquet(features_path)
                st.info(f"Features loaded: {len(features)} observations, {len(features.columns)} features")
            else:
                features = None
                st.warning("Features file not found")
        
        # Results section
        results_key = f"results_{selected_algorithm}_{selected_mode}"
        if results_key in st.session_state:
            st.header("📊 Results")
            
            results = st.session_state[results_key]
            
            # Results tabs
            tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Regime Analysis", "Performance", "Download"])
            
            with tab1:
                results_display.display_results_overview(results)
            
            with tab2:
                results_display.display_regime_analysis(results)
            
            with tab3:
                results_display.display_performance_metrics(results)
            
            with tab4:
                results_display.create_download_links(results, selected_algorithm, selected_mode)
        
        # Model comparison section
        st.header("⚖️ Model Comparison")
        
        # Collect all results for comparison
        all_results = []
        for key, value in st.session_state.items():
            if key.startswith("results_") and isinstance(value, dict):
                all_results.append(value)
        
        if len(all_results) >= 2:
            # Compare models
            comparison_df = model_comparison.compare_models(all_results, features)
            
            if not comparison_df.empty:
                model_comparison.display_comparison_table(comparison_df)
                model_comparison.display_comparison_charts(comparison_df)
                model_comparison.display_ranking(comparison_df)
                
                # Export comparison
                col1, col2 = st.columns(2)
                with col1:
                    csv_data = model_comparison.export_comparison_data(comparison_df, "csv")
                    st.download_button(
                        "📥 Download Comparison (CSV)",
                        csv_data,
                        file_name="model_comparison.csv",
                        mime="text/csv"
                    )
                
                with col2:
                    report = model_comparison.generate_comparison_report(comparison_df)
                    st.download_button(
                        "📄 Download Report (TXT)",
                        report,
                        file_name="comparison_report.txt",
                        mime="text/plain"
                    )
        else:
            st.info("Run at least 2 different algorithms to enable comparison")
    
    else:
        # Welcome message
        st.info("👈 Please select an algorithm and mode from the sidebar to get started")
        
        # Show available algorithms
        st.subheader("Available Algorithms")
        available_algorithms = execution_engine.get_available_algorithms()
        
        for algo in available_algorithms:
            config = algorithm_selector._get_algorithm_config(algo)
            if config:
                st.markdown(f"**{config['name']}**: {config['description']}")
    
    # Footer with regime info
    st.markdown("---")
    catalog = load_regime_catalog()
    show_regime_info(catalog)


if __name__ == "__main__":
    main()
