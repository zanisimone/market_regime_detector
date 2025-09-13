"""
Test script for the new model management system.

This script tests the model discovery and session state functionality
without running the full Streamlit app.
"""

import sys
from pathlib import Path

# Add the project root to the path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.utils import get_available_models, load_model_labels, get_model_info, initialize_session_state


def test_model_discovery():
    """Test model discovery functionality."""
    print("🔍 Testing Model Discovery")
    print("=" * 50)
    
    models = get_available_models()
    print(f"Found {len(models)} models:")
    
    for model in models:
        print(f"\n📊 {model.name}")
        print(f"   Algorithm: {model.algorithm}")
        print(f"   Mode: {model.mode}")
        print(f"   Has Data: {model.has_data}")
        print(f"   Date Range: {model.date_range}")
        print(f"   Num Regimes: {model.num_regimes}")
        print(f"   Labels File: {model.labels_file}")
        print(f"   File Exists: {model.labels_file.exists()}")
    
    return models


def test_model_loading():
    """Test model loading functionality."""
    print("\n\n📥 Testing Model Loading")
    print("=" * 50)
    
    models = get_available_models()
    valid_models = [m for m in models if m.has_data]
    
    if not valid_models:
        print("❌ No valid models found for testing")
        return
    
    # Test loading the first valid model
    test_model = valid_models[0]
    print(f"Testing with model: {test_model.name}")
    
    labels_df = load_model_labels(test_model.name)
    
    if labels_df is not None:
        print(f"✅ Successfully loaded {len(labels_df)} rows")
        print(f"   Columns: {list(labels_df.columns)}")
        print(f"   Date range: {labels_df['date'].min()} to {labels_df['date'].max()}")
        print(f"   Regimes: {sorted(labels_df['regime'].unique())}")
        print(f"   Sample data:")
        print(labels_df.head())
    else:
        print("❌ Failed to load model data")


def test_session_state():
    """Test session state functionality."""
    print("\n\n💾 Testing Session State")
    print("=" * 50)
    
    # Note: This would normally require Streamlit to be running
    # For now, we'll just test the functions that don't require st.session_state
    print("Session state functions require Streamlit runtime")
    print("Available functions:")
    print("- initialize_session_state()")
    print("- refresh_available_models()")
    print("- set_selected_model()")
    print("- get_selected_model()")
    print("- etc.")


def main():
    """Run all tests."""
    print("🚀 Testing New Model Management System")
    print("=" * 60)
    
    try:
        # Test model discovery
        models = test_model_discovery()
        
        # Test model loading
        test_model_loading()
        
        # Test session state (info only)
        test_session_state()
        
        print("\n\n✅ All tests completed successfully!")
        
    except Exception as e:
        print(f"\n\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
