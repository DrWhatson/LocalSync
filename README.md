# LocalSync: Physical Synchrotron Emission Model

**WP4.1 RadioForegroundsPlus Deliverable**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

LocalSync is a Python package for modeling Galactic synchrotron emission using physical synchrotron models (JP, KP, CI) with moment expansion for fast, interpretable fitting.

## Overview

Traditional synchrotron foreground models use empirical power laws with spatially varying spectral indices. LocalSync replaces these with **physical models** based on cosmic ray electron (CRE) physics:

- **JP (Jaffe-Perola)**: Impulsive injection with pitch angle scattering
- **KP (Kardashev-Pacholczyk)**: Impulsive injection, no scattering
- **CI (Continuous Injection)**: Steady injection with optional remnant phase

### Key Innovation: Moment Expansion

Using the SpyDust formalism, LocalSync linearizes physical models around pivot parameters:

```
SED(ν; s, ν_b) ≈ SED₀(ν) + (s-s₀)·∂SED/∂s + (log ν_b - log ν_{b,0})·∂SED/∂log ν_b
```

**Result**: 1000× speedup over grid search, with physically interpretable parameters.

## Installation

```bash
git clone https://github.com/radioforegroundsplus/localsync.git
cd localsync
pip install -e .
```

## Quick Start

```python
import numpy as np
from localsync import fit_spectrum

# Frequencies (Hz)
freqs = np.array([23, 30, 44, 70, 100, 143, 217, 353]) * 1e9

# Your synchrotron data
data = your_synchrotron_spectrum

# Fit with moment expansion
result = fit_spectrum(
    data, freqs,
    model_type='JP',
    method='linear'
)

print(f"Injection index: s = {result['s']:.2f}")
print(f"Break frequency: ν_b = {result['nu_b']:.2e} Hz")
print(f"Reduced χ²: {result['chi2_reduced']:.2f}")
```

## Features

- **Physical Models**: JP, KP, CI-on, CI-off synchrotron models
- **Moment Expansion**: Fast linear fitting via precomputed derivatives
- **Spatial Fitting**: Fit entire sky cubes with `SpatialFitter`
- **Tomography**: 3D Local Bubble reconstruction (in development)
- **Hybrid ML**: Physics-informed neural network support

## Documentation

See `examples/` directory for:
- `example_basic_fit.py`: Simple spectrum fitting
- `example_spatial_fit.py`: Fitting sky maps
- `example_tomography.py`: 3D Local Bubble reconstruction

## WP4.1 Context

LocalSync is the deliverable for **RadioForegroundsPlus WP4.1**: "Modelling the diffuse Galactic synchrotron emission."

**Current Status (M24)**: Core physics models and moment expansion implemented. Testing on WP3.2 synchrotron maps.

**Target (M36)**: Validated 3D synchrotron model for Local Bubble and Galactic background.

## Citation

If you use LocalSync, please cite:

```
RadioForegroundsPlus WP4.1 (2024). LocalSync: Physical Synchrotron Emission Model.
Based on moment expansion from SpyDust (Zhang et al. 2024) and
synchrotron models from Turner & Quici (2018).
```

## License

MIT License - see LICENSE file.

## Contact

WP4.1 Lead: MAN (University of Manchester)
Project: RadioForegroundsPlus
