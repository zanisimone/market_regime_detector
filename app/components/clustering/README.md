# Clustering Components

This directory contains modular components for the clustering functionality in the Market Regime Detector Streamlit app.

## Components Overview

### Core Components

- **`algorithm_selector.py`** - Algorithm selection and configuration interface
- **`parameter_forms.py`** - Dynamic parameter form generation
- **`execution_engine.py`** - Unified execution engine for all algorithms
- **`results_display.py`** - Comprehensive results visualization
- **`model_comparison.py`** - Model comparison and benchmarking tools

### Utility Components

- **`validation.py`** - Parameter validation system
- **`utils.py`** - Common utility functions

## Architecture

The clustering components follow a modular architecture with clear separation of concerns:

```
AlgorithmSelector → ParameterForm → ExecutionEngine → ResultsDisplay
                                                      ↓
                                              ModelComparison
```

### Data Flow

1. **Algorithm Selection**: User selects algorithm and mode
2. **Parameter Configuration**: Dynamic form generation based on algorithm
3. **Validation**: Comprehensive parameter validation
4. **Execution**: Unified execution with progress tracking
5. **Results Display**: Multi-tab visualization of results
6. **Model Comparison**: Compare multiple models side-by-side

## Usage

### Basic Usage

```python
from app.components.clustering import (
    AlgorithmSelector,
    ParameterForm,
    ClusteringExecutionEngine,
    ResultsDisplay
)

# Initialize components
selector = AlgorithmSelector()
form = ParameterForm()
engine = ClusteringExecutionEngine()
display = ResultsDisplay()

# Select algorithm
algorithm, mode = selector.render_algorithm_selection()

# Configure parameters
params = form.render_parameter_form(algorithm, mode)

# Execute algorithm
success, results = engine.execute_algorithm(algorithm, mode, params)

# Display results
display.display_results_overview(results)
```

### Advanced Usage

```python
# Model comparison
comparison = ModelComparison()
results_list = [results1, results2, results3]
comparison_df = comparison.compare_models(results_list)
comparison.display_comparison_table(comparison_df)
```

## Configuration

### Algorithm Configuration

Algorithms are configured in `app/configs/algorithms.json`:

```json
{
  "kmeans": {
    "name": "K-Means",
    "description": "Hard clustering with distance-based assignment",
    "parameters": {
      "n_clusters": {"type": "int", "min": 2, "max": 10, "default": 3},
      "random_state": {"type": "int", "min": 0, "max": 10000, "default": 42}
    },
    "modes": ["split", "rolling"],
    "script": "scripts.run_kmeans_split",
    "rolling_script": "scripts.run_kmeans_rolling"
  }
}
```

### Parameter Types

- **`int`**: Integer input with min/max validation
- **`float`**: Float input with min/max validation
- **`select`**: Dropdown selection from predefined options
- **`bool`**: Checkbox input
- **`text`**: Text input

## Features

### Algorithm Support

- **K-Means**: Hard clustering with distance-based assignment
- **GMM**: Soft clustering with probabilistic assignment
- **HMM**: Sequential modeling with temporal dependencies

### Training Modes

- **Split Mode**: Train once on historical data, apply to full dataset
- **Rolling Mode**: Retrain on moving window for time-adaptive regimes

### Validation

- **Parameter Validation**: Type checking, range validation, required field validation
- **Date Validation**: Date format validation, logical date range validation
- **File Validation**: File existence, format validation, data quality checks

### Visualization

- **Overview**: Basic statistics and regime distribution
- **Regime Analysis**: Timeline, transitions, detailed statistics
- **Performance**: Model quality metrics and execution information
- **Comparison**: Side-by-side model comparison with rankings

### Export

- **CSV Export**: Download results in CSV format
- **JSON Export**: Download results in JSON format
- **Report Generation**: Text-based comparison reports

## Error Handling

The components include comprehensive error handling:

- **Validation Errors**: Clear error messages for invalid inputs
- **Execution Errors**: Detailed error reporting for failed executions
- **File Errors**: Graceful handling of missing or corrupted files
- **Timeout Handling**: Automatic timeout for long-running operations

## Performance

### Optimization Features

- **Progress Tracking**: Real-time progress updates for long operations
- **Caching**: Session state caching for validated parameters
- **Lazy Loading**: On-demand loading of large datasets
- **Vectorized Operations**: Efficient data processing

### Memory Management

- **Chunked Processing**: Large datasets processed in chunks
- **Garbage Collection**: Automatic cleanup of temporary objects
- **Resource Monitoring**: Memory usage tracking and optimization

## Testing

### Unit Tests

```python
# Test individual components
python test_clustering_components.py
```

### Integration Tests

```python
# Test full pipeline
from app.components.clustering import *
# Run end-to-end tests
```

## Dependencies

### Required

- `streamlit` - UI framework
- `pandas` - Data manipulation
- `numpy` - Numerical operations
- `matplotlib` - Plotting
- `seaborn` - Statistical visualization
- `scikit-learn` - Machine learning metrics

### Optional

- `plotly` - Interactive visualizations
- `hmmlearn` - Hidden Markov Models
- `joblib` - Model serialization

## Contributing

### Adding New Algorithms

1. Add algorithm configuration to `algorithms.json`
2. Implement algorithm-specific script
3. Add algorithm-specific validation rules
4. Update documentation

### Adding New Features

1. Create new component or extend existing
2. Add comprehensive tests
3. Update documentation
4. Add example usage

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure project root is in Python path
2. **Validation Errors**: Check parameter types and ranges
3. **Execution Errors**: Verify script paths and dependencies
4. **Display Errors**: Check data format and column names

### Debug Mode

Enable debug mode by setting environment variable:

```bash
export STREAMLIT_DEBUG=true
```

## License

This project is licensed under the MIT License - see the main project LICENSE file for details.
