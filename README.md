# AutoOptimizedKNN: Intelligent K-Nearest Neighbors with Automatic Optimization

## Overview

**AutoOptimizedKNN** is an advanced K-Nearest Neighbors implementation that automatically analyzes input data to select the most efficient scaling method, distance metric, and search structure. Unlike traditional KNN implementations, this library intelligently adapts to your data characteristics to provide optimal performance without manual tuning.

## Key Features

🧠 **Intelligent Auto-Optimization**
- Automatic Data Profiling: Analyzes data shape, distribution, skewness, and feature types
- Adaptive Algorithm Selection: Chooses between Brute Force, KD-Tree, Ball Tree, and HNSW based on data characteristics
- Smart Distance Metric Selection: Automatically selects Euclidean, Manhattan, Minkowski, or Mahalanobis distances
- Adaptive Preprocessing: Selects optimal scaling (Standard, Robust, MinMax, Yeo-Johnson) based on data characteristics

🚀 **Performance Optimizations**
- Vectorized Operations: NumPy-based vectorization for all distance calculations
- Multiple Search Strategies: Support for efficient nearest neighbor search algorithms
- Caching System: Caches repeated distance computations for improved efficiency
- Parallel Processing: Optional parallelization for large datasets

🔧 **Advanced Capabilities**
- Mixed Data Types: Support for continuous and categorical features
- Outlier Detection & Handling: Built-in outlier detection and multiple handling strategies
- Dimensionality Reduction: Automatic PCA recommendation for high-dimensional data
- Configuration Overrides: YAML-based configuration for manual tuning when needed

📊 **Monitoring & Visualization**
- Runtime Logging: Comprehensive logging for performance monitoring
- Data Visualization: Basic visualization tools for understanding data distributions
- Analysis Reports: JSON export of data characteristics and optimization decisions

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/AutoOptimizedKNN.git
cd AutoOptimizedKNN

# Install dependencies
pip install numpy pandas scipy matplotlib pyyaml

# Install the package
pip install -e .
```

## Quick Start

```python
from autoknn import SmartKNN
import numpy as np

# Generate sample data
np.random.seed(42)
X_train = np.random.randn(1000, 10)
y_train = np.random.randint(0, 2, 1000)
X_test = np.random.randn(200, 10)

# Initialize and train
knn = SmartKNN(k=5)
knn.fit(X_train, y_train)

# Make predictions
predictions = knn.predict(X_test)

# Get optimization report
report = knn.get_optimization_report()
print(f"Selected strategy: {report['search_strategy']}")
print(f"Distance metric: {report['distance_metric']}")
```

## Project Architecture

```
AutoOptimizedKNN/
├── autoknn/                    # Main package
│   ├── __init__.py
│   ├── data_profiler.py       # DataProfiler - Intelligent data analysis
│   ├── distance_engine.py     # DistanceEngine - Vectorized distance metrics
│   ├── search_strategy.py     # SearchStrategyFactory - Search algorithms
│   ├── preprocess_flow.py     # PreProcessFlow - Automated preprocessing
│   ├── scalers.py             # Custom scaler implementations
│   ├── smart_knn.py           # SmartKNN - Main interface
│   ├── config.py              # Configuration handling
│   ├── exceptions.py          # Custom exceptions
│   └── utils.py               # Helper functions
├── tests/                     # Comprehensive test suite
├── examples/                  # Usage examples
├── config/                    # Configuration files
├── docs/                      # Documentation
└── benchmarks/               # Performance benchmarks
```

## Core Components

1. **DataProfiler (Intelligent Data Analysis)**

   The "brain" of the system that analyzes data characteristics:
   - Data type detection (continuous, categorical, binary)
   - Distribution analysis (skewness, kurtosis)
   - Sparsity and outlier detection
   - Feature correlation analysis
   - Dimensionality assessment

   ```python
   from autoknn import DataProfiler

   profiler = DataProfiler()
   plan = profiler.analyze(X_train, y_train)

   print(f"Recommended search: {plan.search_strategy}")
   print(f"Recommended distance: {plan.distance_metric}")
   print(f"Recommended scaling: {plan.scaling_method}")
   ```

2. **DistanceEngine (Vectorized Distance Metrics)**

   Efficient, vectorized implementations of distance metrics:
   - Euclidean distance
   - Manhattan distance
   - Minkowski distance (with configurable p)
   - Mahalanobis distance (accounts for feature correlations)

   ```python
   from autoknn.distance_engine import DistanceEngine

   engine = DistanceEngine(metric='euclidean')
   distances = engine.compute_pairwise(X1, X2)
   ```

3. **SearchStrategyFactory (Optimized Search)**

   Multiple search strategies for different scenarios:
   - Brute Force: Simple, works for all cases
   - KD-Tree: Efficient for low-dimensional data (d ≤ 20)
   - Ball Tree: Better for moderate dimensions (d ≤ 50)
   - HNSW: Hierarchical Navigable Small World for high-dimensional/large datasets

4. **PreProcessFlow (Adaptive Preprocessing)**

   Automatic preprocessing based on data characteristics:
   - Scaling: StandardScaler, RobustScaler, MinMaxScaler, Yeo-Johnson
   - Outlier handling: removal, capping, or ignoring
   - Categorical encoding: label or one-hot encoding

## Configuration

### Automatic Mode (Recommended)

```python
# The system analyzes your data and makes optimal choices
knn = SmartKNN(k=5, auto_optimize=True)
knn.fit(X_train, y_train)
```

### Manual Override Mode

```yaml
# config/custom_config.yaml
search_strategy: ball_tree
distance_metric: manhattan
scaling_method: robust
outlier_handling: cap
categorical_encoding: onehot
```

```python
# Use custom configuration
knn = SmartKNN(k=5, config_path='config/custom_config.yaml')
```

## Advanced Usage

### Mixed Data Types

```python
# The system automatically handles mixed data types
X_mixed = np.column_stack([
    np.random.randn(1000, 5),  # Continuous features
    np.random.randint(0, 5, (1000, 3))  # Categorical features
])

knn = SmartKNN(k=5)
knn.fit(X_mixed, y_train)
```

### Outlier Handling

```python
# Enable outlier detection and handling
knn = SmartKNN(
    k=5,
    handle_outliers=True,
    outlier_method='isolation_forest'
)
```

## Performance Monitoring

```python
import logging

# Enable detailed logging
logging.basicConfig(level=logging.DEBUG)

knn = SmartKNN(k=5)
knn.fit(X_train, y_train)

# Get performance metrics
metrics = knn.get_performance_metrics()
print(f"Training time: {metrics['training_time']:.2f}s")
print(f"Average query time: {metrics['avg_query_time']:.4f}s")
```

## Performance Benchmarks

| Data Characteristics             | Recommended Strategy | Expected Speedup        |
|----------------------------------|----------------------|-------------------------|
| n < 100, d < 3                   | Brute Force          | Baseline                |
| 100 ≤ n < 10k, d ≤ 20            | KD-Tree              | 5-10x faster            |
| n > 10k, 20 < d ≤ 50             | Ball Tree            | 10-50x faster           |
| n > 100k, d > 50                 | HNSW                 | 50-100x faster          |
| High sparsity (>70%)             | Brute Force          | Optimized for sparsity   |

## API Reference

### SmartKNN Class

```python
class SmartKNN:
    def __init__(self, k=5, auto_optimize=True, config_path=None, **kwargs):
        """
        Initialize the AutoOptimizedKNN model.
        
        Args:
            k: Number of neighbors (default: 5)
            auto_optimize: Enable automatic optimization (default: True)
            config_path: Path to configuration file (optional)
            **kwargs: Additional parameters
        """
    
    def fit(self, X, y):
        """
        Train the model with automatic optimization.
        
        Args:
            X: Training data (n_samples, n_features)
            y: Target values
        """
    
    def predict(self, X):
        """
        Make predictions on new data.
        
        Args:
            X: Test data (n_samples, n_features)
        
        Returns:
            Predictions for each sample
        """
    
    def predict_proba(self, X):
        """
        Return probability estimates for classification.
        
        Args:
            X: Test data
        
        Returns:
            Probability estimates
        """
    
    def get_optimization_report(self):
        """
        Get detailed report of optimization decisions.
        
        Returns:
            Dictionary with optimization details
        """
```

### DataProfiler Class

```python
class DataProfiler:
    def analyze(self, X, y=None):
        """
        Analyze data and generate optimization plan.
        
        Returns:
            OptimizationPlan object with recommendations
        """
    
    def get_summary(self):
        """
        Get comprehensive analysis summary.
        
        Returns:
            Dictionary with data characteristics
        """
    
    def save_analysis(self, filepath):
        """
        Save analysis results to JSON file.
        """
```

## Examples

See the `examples/` directory for complete examples:
1. Basic Classification: `examples/basic_classification.py`
2. Regression: `examples/regression.py`
3. Mixed Data Types: `examples/mixed_data_types.py`
4. Large Dataset: `examples/large_dataset.py`
5. Custom Configuration: `examples/custom_config.py`

## Development

### Setting Up Development Environment

```bash
# Clone and setup
git clone https://github.com/yourusername/AutoOptimizedKNN.git
cd AutoOptimizedKNN

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements-dev.txt

# Install in development mode
pip install -e .
```

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test module
pytest tests/test_data_profiler.py -v

# Run with coverage
pytest --cov=autoknn tests/
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## Roadmap

### Phase 1 (Current) ✅
- DataProfiler implementation
- Basic distance metrics
- Simple search strategies
- Automatic optimization logic

### Phase 2 (Next) 🔄
- Advanced search algorithms (HNSW)
- Parallel processing support
- GPU acceleration (optional)
- Streaming data support

### Phase 3 (Planned) 📅
- Integration with scikit-learn API
- Distributed computing support
- Advanced visualization tools
- Web-based configuration interface

## Performance Tips

1. For small datasets (n < 1000): Use default settings
2. For high-dimensional data (d > 100): Enable PCA in configuration
3. For very large datasets (n > 1M): Use HNSW with batch prediction
4. For mixed data types: Ensure categorical features are properly encoded

## Citation

If you use AutoOptimizedKNN in your research, please cite:

```bibtex
@software{autoknn2025,
  title = {AutoOptimizedKNN: Intelligent K-Nearest Neighbors with Automatic Optimization},
  author = {Zwiswa Muridili},
  year = {2025},
  url = {https://github.com/JepStar990/AutoOptimizedKNN}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

- 📖 Documentation: ReadTheDocs
- 🐛 Issues: GitHub Issues
- 💬 Discussions: GitHub Discussions
- 📧 Email: zwiswamuridili990@gmail.com

## Acknowledgments

- Inspired by scikit-learn's KNeighborsClassifier
- HNSW algorithm based on hnswlib
- Distance metrics optimized using NumPy vectorization
