# Ordered Tree-Wasserstein Distance

This project implements methods for measuring similarity between temporal sequences using Tree-Wasserstein Distance.

## Project Overview

OTWD (Ordered Tree-Wasserstein Distance) provides tools for analyzing and comparing temporal sequences using advanced mathematical techniques combining optimal transport theory and dynamic programming. The project includes implementations of various algorithms and methods for sequence comparison, particularly useful in time series analysis and pattern recognition.

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
├── examples/                   # Jupyter notebook examples
│   ├── custom_functions.ipynb
│   ├── example.ipynb
│   ├── plot.ipynb
│   └── testing.ipynb
└── src/                       # Source code
	├── dtw.py
	├── utilities.py
	├── bs/                    # Base implementation
	│   └── normalize.py
	├── gow/                   # Generalized Ordered Wasserstein
	│   └── utilities.py
	└── otwd_star/            # OTWD* implementation
```

## Dependencies

The project requires the following Python packages:

- numpy
- scikit-learn
- POT (Python Optimal Transport)
- aeon
- joblib
- tslearn
- seaborn
- matplotlib
- tensorflow (optional)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/OTWD.git
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Examples

The `examples/` directory contains several Jupyter notebooks demonstrating various use cases:

- `experiment.ipynb`: Run the main experiment on the UCR dataset
- `custom_functions.ipynb`: Examples of using custom functions (for GOW)
- `plot.ipynb`: Visualization examples
- `testing.ipynb`: Testing and validation examples

## Features

- Implementation of various optimal transport algorithms
- Support for multiple distance metrics
- Built-in dataset loaders for UCR and Human Action datasets
- K-NN classifier implementation
- Visualization tools
- Custom function support for distance calculations

## Dataset Support

### UCR Datasets
- BasicMotions
- BME
- Chinatown
- DistalPhalanxTW
- ItalyPowerDemand

### Human Action Datasets
- MSRAction3D
- MSRDailyActivity3D
- SpokenArabicDigit
- Weizmann

### UCR Datasets from tslearn
- Import from tslearn

## Development

The source code is organized into several modules:

- `src/dtw.py`: Dynamic Time Warping implementation
- `src/utilities.py`: General utility functions
- `src/bs/`: OTWD implementation module
- `src/gow/`: Generalized Ordered Wasserstein implementation
- `src/otwd_star/`: CTWD algorithm implementation

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

For questions and support, please open an issue in the project's issue tracker.
