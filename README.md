# LocalSync: Physical Synchrotron Emission Model

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
git clone https://github.com/DrWhatson/LocalSync.git
cd LocalSync
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
- **Spatial Fitting**: Fit entire sky cubes with spatial regularization
- **Tomography**: 3D Local Bubble reconstruction (in development)
- **Light Box**: Iterative fitting of foreground (LB) + background (Galaxy)

## Documentation

See `examples/` directory for:
- `example_basic_fit.py`: Simple spectrum fitting
- `example_tomography.py`: 3D Local Bubble reconstruction
- `example_lightbox.py`: Iterative foreground/background fitting

See `docs/TOMOGRAPHY.md` for detailed tomography documentation.

## Citation

If you use LocalSync, please cite:

```
LocalSync: Physical Synchrotron Emission Model.
Based on moment expansion from SpyDust (Zhang et al. 2024) and
synchrotron models from Turner & Quici (2018).
```

## References

- **SpyDust**: Zhang et al. (2024) - Moment expansion formalism
- **synchrofit**: Turner & Quici (2018) - JP/KP/CI synchrotron models
- **ONeill2024**: O'Neill et al. (2024) - Local Bubble 3D magnetic field

## License

MIT License - see LICENSE file.

## Contact

For questions and contributions, please open an issue on GitHub.
