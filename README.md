# Ordered Tree Sliced Wasserstein (OTSW)

This project implements multiple methods for measuring similarity between temporal sequences using various Optimal Transport-based distances, including Tree-Wasserstein Distance and other state-of-the-art approaches.

## Project Overview

OTSW provides comprehensive tools for analyzing and comparing temporal sequences using advanced mathematical techniques combining optimal transport theory and dynamic programming. The project includes implementations of various distance algorithms for time series analysis, clustering, and pattern recognition.

### Implemented Distance Methods

- **OTSW (Ordered Tree Sliced Wasserstein)** - Tree-based optimal transport with temporal constraints
- **ASW (Auto-weighted Sequential Wasserstein)** - Automatically balanced multi-component distance
- **TAOT (Time-Adaptive Optimal Transport)** - Time-aware OT with entropic regularization
- **POW (Partial Ordered Wasserstein)** - Bandwidth-constrained temporal matching
- **TCOT (Temporally Coupled Optimal Transport)** - Coupled spatial and temporal distance
- **OPW (Ordered Preserving Wasserstein)** - GPU-accelerated order-preserving distance
- **DTW (Dynamic Time Warping)** - Classic time series alignment
- **GOW (Generalized Ordered Wasserstein)** - Generalized version with Sinkhorn

## Directory Structure

```
├── data/
│   ├── Human_Actions/          # Human action datasets
│   │   ├── MSRAction3D/
│   │   ├── MSRDailyActivity3D/
│   │   ├── SpokenArabicDigit/
│   │   └── Weizmann/
│   └── UCR/                    # UCR Time Series datasets
│       ├── BasicMotions/
│       ├── BME/
│       ├── Chinatown/
│       ├── DistalPhalanxTW/
│       └── ItalyPowerDemand/
├── experiment/                 # Experiment notebooks
│   ├── kmedoids.ipynb         # K-Medoids clustering experiments
│   └── knn.ipynb              # K-NN classification experiments
└── src/                       # Source code
    ├── dtw.py                 # Dynamic Time Warping
    ├── utilities.py           # Dataset loaders and evaluation
    ├── bs/                    # Baseline methods
    │   ├── __init__.py
    │   └── normalize.py       # Normalization utilities
    ├── gow/                   # Generalized Ordered Wasserstein
    │   ├── __init__.py
    │   └── utilities.py
    └── otsw/                  # OTSW implementation
        └── __init__.py
```

## Dependencies

The project requires the following Python packages:

### Core Dependencies
- **numpy** - Numerical computing
- **scipy** - Scientific computing and sparse matrices
- **scikit-learn** - Machine learning utilities
- **sklearn-extra** - K-Medoids clustering
- **POT (Python Optimal Transport) >= 0.9.3** - Optimal transport algorithms
- **joblib** - Data serialization

### Time Series and Datasets
- **aeon** - Time series machine learning
- **tslearn** - Time series datasets (UCR/UEA)

### Visualization
- **seaborn** - Statistical visualization
- **matplotlib** - Plotting library

### GPU Support (Optional)
- **cupy** - GPU-accelerated computing for OPW, ASW (requires CUDA)
- **tensorflow** - Deep learning framework (optional)

### Other
- **pandas** - Data manipulation and CSV export

## Installation

1. Clone the repository:
```bash
git clone https://github.com/HoangLeMinh0809/OTWD.git
cd OTWD
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. (Optional) For GPU support with OPW and ASW:
```bash
pip install cupy-cuda11x  # Replace with your CUDA version
```

## Quick Start

### K-Medoids Clustering

Run clustering experiments using [experiment/kmedoids.ipynb](experiment/kmedoids.ipynb):

1. Open the notebook
2. Run all cells to define functions
3. In **Cell 9**, change the `ALG` variable:
   ```python
   ALG = "OTSW"  # Options: "OTSW", "OPW", "TCOT", "POW", "ASW", "TAOT"
   ```
4. Execute the cell to run clustering and save results to CSV

**Output:** Results include ACC (accuracy), NMI (Normalized Mutual Information), and timing metrics.

### K-NN Classification

Use [experiment/knn.ipynb](experiment/knn.ipynb) for classification tasks:

```python
from utilities import run_knn

# Run K-NN with DTW
alg = "DTW"
datatype = "UCR_TSL"
run_knn('Weizmann', datatype, alg)
```

## Usage Examples

### Using OTSW (Tree-based Distance)

```python
from src.otsw import build_otsw_tamle, otsw_between_series_fast
import numpy as np

# Prepare sequences (list of arrays or 3D array)
sequences = [np.random.randn(50, 3), np.random.randn(60, 3)]

# Build OTSW model
model = build_otsw_tamle(
    sequences,
    lam_time=5.0,      # Time regularization
    leaf_size=16,      # Tree leaf size
    max_depth=20,      # Max tree depth
    seed=0
)

# Compute distance between series 0 and 1
distance = otsw_between_series_fast(model, 0, 1)
print(f"OTSW distance: {distance}")
```

### Using ASW/TAOT/TCOT Distances

```python
import numpy as np
from asw import asw_distance  # or taot_distance, tcot_distance_series

X = np.random.randn(100, 5)  # Series 1: 100 timesteps, 5 features
Y = np.random.randn(120, 5)  # Series 2: 120 timesteps, 5 features

# Compute distance
dist = asw_distance(X, Y, lam=10.0, auto_weight=True)
print(f"ASW distance: {dist}")
```

### Loading Datasets

```python
from src.utilities import load_ucr_dataset, load_human_action_dataset

# Load UCR dataset via tslearn
from tslearn.datasets import UCR_UEA_datasets
ucr = UCR_UEA_datasets()
X_train, y_train, X_test, y_test = ucr.load_dataset("BasicMotions")

# Load Human Action dataset
X_tr, y_tr, X_te, y_te = load_human_action_dataset(
    "data/Human_Actions", 
    "Weizmann"
)
```

## Distance Methods Comparison

| Method | Description | GPU Support | Key Parameters |
|--------|-------------|-------------|----------------|
| **OTSW** | Tree-based OT with temporal ordering | ❌ | `lam_time`, `leaf_size`, `max_depth` |
| **ASW** | Auto-weighted spatial + order + structure | ✅ | `lam`, `auto_weight` |
| **TAOT** | Time-adaptive OT with entropic reg. | ✅ | `lam`, `w` (time weight) |
| **POW** | Partial OT with bandwidth constraint | ✅ | `lam`, `bandwidth`, `lam_order` |
| **TCOT** | Temporally coupled OT | ✅ | `lambda_pos`, `reg` |
| **OPW** | Order-preserving with kernel | ✅ (required) | `lambda1`, `lambda2`, `sigma` |
| **DTW** | Classic dynamic time warping | ❌ | `sakoe_chiba_radius` |
| **GOW** | Generalized ordered Wasserstein | ❌ | `lam`, `scale` |

### Method Recommendations

- **For ragged/variable-length sequences**: OTSW, ASW, TAOT, POW, TCOT
- **For GPU acceleration**: OPW (required), ASW, TAOT, POW, TCOT (optional)
- **For clustering**: OTSW, ASW (best balance of accuracy and speed)
- **For interpretability**: DTW (classic), OTSW (tree structure)
- **For large-scale datasets**: OPW (GPU), OTSW (efficient tree-based)

## Notebooks and Experiments

### Experiment Notebooks
- **[experiment/kmedoids.ipynb](experiment/kmedoids.ipynb)** - K-Medoids clustering with multiple distance methods
  - Comprehensive documentation for each method
  - Easy method switching via single variable
  - Automatic result export to CSV
  - Supports all implemented distances

- **[experiment/knn.ipynb](experiment/knn.ipynb)** - K-NN classification experiments
  - UCR and Human Action dataset support
  - Distance matrix precomputation
  - Accuracy evaluation

### Tutorial Examples
- **[examples/custom_functions.ipynb](examples/custom_functions.ipynb)** - Custom GOW functions
- **[examples/example.ipynb](examples/example.ipynb)** - Basic usage examples
- **[examples/plot.ipynb](examples/plot.ipynb)** - Visualization tools
- **[examples/testing.ipynb](examples/testing.ipynb)** - Testing utilities

### UCR Time Series Archive
Access via `tslearn.datasets.UCR_UEA_datasets()`:
- ArrowHead, BasicMotions, BeetleFly, CBF
- Chinatown, CinCECGTorso, DiatomSizeReduction
- GunPointAgeSpan, GunPointMaleVersusFemale, GunPointOldVersusYoung
- Ham, InsectEPGRegularTrain, ItalyPowerDemand
- Meat, MelbournePedestrian, MoteStrain
- OliveOil, Plane, SmoothSubspace
- And 100+ more datasets from UCR/UEA archive

### Human Action Datasets
Located in `data/Human_Actions/`:
- **MSRAction3D** - 3D skeleton-based action recognition
- **MSRDailyActivity3D** - Daily activity recognition
- **SpokenArabicDigit** - Speech recognition time series
- **Weizmann** - Human action sequences

## Performance Tips

1. **GPU Acceleration**: Install CuPy for 10-100x speedup with OPW, ASW, TAOT, POW, TCOT
   ```bash
   pip install cupy-cuda11x
   ```

2. **OTSW Parameters**:
   - Decrease `leaf_size` (4-16) for better accuracy, increase for speed
   - Increase `lam_time` (5-20) for stronger temporal alignment
   - Adjust `max_depth` based on dataset size

3. **Memory Management**:
   - For large datasets (>1000 sequences), compute distance matrices in batches
   - Use `dtype=float32` instead of `float64` to reduce memory usage

4. **Parallelization**: 
   - Use `joblib.Parallel` for batch distance computations
   - GPU methods handle batches automatically

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-method`)
3. Commit your changes with clear messages
4. Add tests and documentation
5. Submit a Pull Request

### Development Guidelines
- Follow NumPy docstring conventions
- Add type hints where applicable
- Include unit tests for new methods
- Update README.md with new features

## License

This project is open source. Please check the repository for license details.

## Contact

- **GitHub Issues**: [Report bugs or request features](https://github.com/HoangLeMinh0809/OTWD/issues)
- **Repository**: https://github.com/HoangLeMinh0809/OTWD

## Acknowledgments

This project builds upon:
- Python Optimal Transport (POT) library
- tslearn for UCR/UEA datasets
- scikit-learn for machine learning utilities
- Research on optimal transport for time series analysis
