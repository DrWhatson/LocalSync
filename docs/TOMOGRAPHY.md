# 3D Local Bubble Tomography with SpySynch

This document describes the tomographic reconstruction of synchrotron emission in the Local Bubble using SpySynch.

## Overview

The tomography solves for 3D distribution of physical parameters:
- **s(r,θ,φ)**: CRE injection index
- **log ν_b(r,θ,φ)**: log break frequency
- **A(r,θ,φ)**: Amplitude

From observed synchrotron maps I(ν,θ,φ), we invert:

```
I(ν, l, b) = ∫₀^∞ ε(ν, r, l, b) dr
```

Using regularized least squares with moment expansion.

## Mathematical Framework

### Forward Model

```
d = P · B · x + noise

where:
- d: observed data [n_freq × n_pix]
- P: projection matrix [n_pix × n_voxels]
- B: moment basis [n_voxels × n_freq × n_params]
- x: parameters [n_voxels × n_params] (flattened)
```

### Regularized Objective

```
min ||d - P·B·x||² + λ_s||L_s·x||² + λ_r||L_r·x||²

where:
- L_s: spatial Laplacian (smoothness on sky)
- L_r: radial derivative (smoothness along LOS)
- λ_s, λ_r: regularization weights
```

### Solution Method

Augmented system:
```
[P·B]     [d]
[λ_s L_s] x = [0]
[λ_r L_r]     [0]
```

Solved iteratively with LSQR or CG.

## Usage

### Basic Tomography

```python
from localsync import LocalBubbleTomography

# Initialize
tomo = LocalBubbleTomography(
    nside=128,
    r_bins=np.linspace(0, 200, 21)  # 0-200 pc, 20 bins
)

# Build projection matrix (do once)
tomo.build_projection_matrix(cache_file='P_matrix.npz')

# Fit
params_3d = tomo.fit(
    data_cube,      # [n_freq, npix] observed maps
    frequencies,    # [n_freq] in Hz
    lambda_s=0.1,   # spatial smoothing
    lambda_r=0.5    # radial smoothing
)

# Results
s_map = params_3d['s']            # [n_radial, npix]
log_nu_map = params_3d['log_nu_b']  # [n_radial, npix]
```

### Regularization Tuning

```python
# Try different weights
for lambda_s in [0.01, 0.1, 0.5, 1.0]:
    for lambda_r in [0.1, 0.5, 1.0, 2.0]:
        params = tomo.fit(data, freqs, lambda_s, lambda_r)
        chi2 = compute_chi2(params, data)
        # Choose best by cross-validation
```

## Physical Structure

### Local Bubble Geometry

```
r < 130 pc:    Interior (turbulent, old electrons)
                → s ≈ 2.8, ν_b ≈ 10-30 GHz

r = 130-200 pc: Wall (compressed, young electrons)
                → s ≈ 2.0-2.3, ν_b ≈ 80-150 GHz

r > 200 pc:     Galactic background (CI model)
                → s ≈ 2.3, ν_b varies
```

### Model Selection

| Region | Model | Physical Reason |
|--------|-------|-----------------|
| LB wall | JP | Impulsive shock acceleration |
| LB interior | JP | Older, cooled population |
| Spiral arms | CI-on | Continuous SNR injection |

## Implementation Details

### Projection Matrix

- Size: [n_pix, n_voxels] where n_voxels = n_radial × n_pix
- Sparse (mostly zeros)
- P[i,j] = path length of voxel j along LOS i
- Precompute and cache (expensive!)

### Spatial Regularization

Laplacian on HEALPix:
```
L[i,i] = N_neighbors(i)
L[i,j] = -1 if j neighbor of i
```

Enforces: smooth variations between adjacent pixels.

### Radial Regularization

Second derivative:
```
L[r] = 2·x[r] - x[r-1] - x[r+1]
```

Enforces: smooth variation with radius (no sharp jumps).

## Performance

| Operation | Time | Memory |
|-----------|------|--------|
| Build P matrix | ~10 min | ~1 GB |
| Spatial L_s | ~1 min | ~500 MB |
| Radial L_r | ~10 sec | ~100 MB |
| Solve (LSQR) | ~5 min | ~2 GB |

Total: ~20 min for nside=128, 20 radial bins.

## Validation

### Check Convergence

```python
# Reduced χ² should be ~1
chi2_red = result['chi2'] / result['dof']
print(f"χ²/dof = {chi2_red:.2f}")

# Residuals should be noise-like
residuals = data - tomo.evaluate_model(params_3d, freqs)
assert np.std(residuals) ≈ noise_level
```

### Physical Checks

```python
# Parameters in expected ranges
assert 2.0 < s < 3.0
assert 8 < log_nu_b < 11

# Smoothness
assert gradient(s) < threshold
assert gradient(log_nu_b) < threshold
```

## Troubleshooting

### Convergence Fails

- **Reduce n_radial**: Try 10 bins instead of 20
- **Increase λ_s**: Force more spatial smoothness
- **Check data**: Ensure no NaNs or infinities

### Unphysical Parameters

- **Increase λ_r**: Force radial smoothness
- **Add prior**: Include physical bounds
- **Fix interior**: Set interior to expected values

### Too Slow

- **Lower nside**: Try nside=64 instead of 128
- **Fewer bins**: Reduce n_radial
- **Cache P**: Save projection matrix to disk

## References

- SpyDust: Zhang et al. 2024 (moment expansion)
- Turner & Quici 2018 (synchrotron models)
- Lallement et al. 2003 (Local Bubble geometry)
