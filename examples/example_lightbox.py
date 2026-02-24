#!/usr/bin/env python
"""
Light Box Iterative Fitting Example

Demonstrates the iterative fitting scheme:
1. Local Bubble: 2D spherical foreground screen
2. Galaxy: 3D cylindrical background

The approach alternates:
- Fix Galaxy, fit LB to residuals
- Fix LB, fit Galaxy to residuals
"""

import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

from localsync import jp_model, ci_model, MomentBasis, LinearFitter
from localsync.lightbox import SphericalForeground, CylindricalBackground


def generate_simple_test_data(nside=32, frequencies=None, noise_level=0.05):
    """
    Generate simple test data with known components.
    """
    if frequencies is None:
        frequencies = np.array([23, 44, 70, 100]) * 1e9

    npix = hp.nside2npix(nside)
    l, b = hp.pix2ang(nside, np.arange(npix), lonlat=True)

    # Local Bubble: high latitude shell
    lb_mask = np.abs(b) > 20
    lb_s = np.where(lb_mask, 2.8, 2.3)  # Steeper at high lat
    lb_nu_b = np.where(lb_mask, 10**10.0, 10**9.5)

    # Galaxy: spiral arm pattern
    gal_s = 2.5 - 0.2 * np.sin(np.radians(l) * 2)

    # Generate emission
    lb_emission = np.zeros((npix, len(frequencies)))
    gal_emission = np.zeros((npix, len(frequencies)))

    for i in range(npix):
        if lb_mask[i]:
            lb_emission[i] = jp_model(frequencies, lb_s[i], lb_nu_b[i])
        gal_emission[i] = ci_model(frequencies, gal_s[i], 10**9.0)

    total = lb_emission + gal_emission

    # Add noise
    noise = noise_level * np.std(total) * np.random.randn(*total.shape)
    data = total + noise

    return data, total, lb_emission, gal_emission, frequencies, lb_mask


def simple_iterative_fit(data, frequencies, nside=32, n_iter=3):
    """
    Simple iterative fitting demonstration.

    Alternates between fitting LB and Galaxy.
    """
    npix = hp.nside2npix(nside)
    n_freq = len(frequencies)

    # Initialize components
    lb_model = np.zeros((npix, n_freq))
    gal_model = np.zeros((npix, n_freq))

    # Create moment basis for fitting
    pivot = {'s': 2.3, 'log_nu_b': 9.5}
    basis = MomentBasis(frequencies, pivot)
    fitter = LinearFitter(basis)

    # Get LB mask
    l, b = hp.pix2ang(nside, np.arange(npix), lonlat=True)
    lb_mask = np.abs(b) > 20

    chi2_history = []

    for iteration in range(n_iter):
        print(f"\n=== Iteration {iteration + 1}/{n_iter} ===")

        # Current total model
        total_model = lb_model + gal_model
        chi2 = np.sum((data - total_model)**2)
        chi2_history.append(chi2)
        print(f"Chi2 before: {chi2:.2e}")

        if iteration % 2 == 0:
            # Fit LB to residuals (data - galaxy)
            print("Fitting Local Bubble...")
            residuals = data - gal_model

            for i in range(npix):
                if not lb_mask[i]:
                    continue
                try:
                    result = fitter.fit(residuals[i])
                    # Simple model: just use the fitted SED directly
                    lb_model[i] = basis.linear_predict([
                        result['amplitude'],
                        result['delta_s'],
                        result['delta_log_nu_b']
                    ])
                except Exception as e:
                    pass  # Keep current value

        else:
            # Fit Galaxy to residuals (data - LB)
            print("Fitting Galaxy...")
            residuals = data - lb_model

            # For Galaxy, fit all pixels
            for i in range(npix):
                try:
                    result = fitter.fit(residuals[i])
                    gal_model[i] = basis.linear_predict([
                        result['amplitude'],
                        result['delta_s'],
                        result['delta_log_nu_b']
                    ])
                except Exception as e:
                    pass

    # Final chi2
    final_model = lb_model + gal_model
    final_chi2 = np.sum((data - final_model)**2)

    return {
        'lb_model': lb_model,
        'gal_model': gal_model,
        'final_model': final_model,
        'chi2_history': chi2_history + [final_chi2],
        'final_chi2': final_chi2,
    }


if __name__ == '__main__':
    print("="*60)
    print("Light Box Iterative Fitting Demo")
    print("="*60)
    print()
    print("This demonstrates the iterative fitting scheme:")
    print("1. Local Bubble: 2D spherical foreground screen")
    print("2. Galaxy: 3D cylindrical background")
    print()
    print("The approach alternates:")
    print("- Fix Galaxy, fit LB to residuals")
    print("- Fix LB, fit Galaxy to residuals")
    print()

    # Generate test data
    print("Generating synthetic test data...")
    frequencies = np.array([23, 44, 70, 100]) * 1e9
    data, truth, lb_true, gal_true, frequencies, lb_mask = generate_simple_test_data(
        nside=32, frequencies=frequencies, noise_level=0.05
    )

    print(f"Data shape: {data.shape}")
    print(f"Frequencies: {frequencies/1e9} GHz")
    print(f"RMS signal: {np.std(truth):.3e}")
    print(f"RMS noise: {np.std(data - truth):.3e}")
    print()
    print("True components:")
    print(f"  LB mean: {np.mean(lb_true):.3e}, RMS: {np.std(lb_true):.3e}")
    print(f"  Galaxy mean: {np.mean(gal_true):.3e}, RMS: {np.std(gal_true):.3e}")
    print()

    # Run iterative fitting
    print("Running iterative fitting...")
    result = simple_iterative_fit(data, frequencies, nside=32, n_iter=3)

    print()
    print("="*60)
    print("Results")
    print("="*60)
    print(f"Chi2 history: {result['chi2_history']}")
    print(f"Final chi2: {result['final_chi2']:.2e}")
    print()

    # Compare to truth
    lb_model = result['lb_model']
    gal_model = result['gal_model']

    print("Component comparison:")
    print(f"  LB model RMS: {np.std(lb_model):.3e} (true: {np.std(lb_true):.3e})")
    print(f"  Galaxy model RMS: {np.std(gal_model):.3e} (true: {np.std(gal_true):.3e})")

    # Correlations
    lb_corr = np.corrcoef(lb_true.flatten(), lb_model.flatten())[0,1]
    gal_corr = np.corrcoef(gal_true.flatten(), gal_model.flatten())[0,1]

    print(f"  LB correlation: {lb_corr:.3f}")
    print(f"  Galaxy correlation: {gal_corr:.3f}")
    print()
    print("Note: This is a simple prototype. Real implementation would:")
    print("  - Use proper 3D cylindrical integration for Galaxy")
    print("  - Add spatial regularization (smoothness priors)")
    print("  - Include polarization (f_turb affects Q/U)")
    print("  - Use more sophisticated convergence criteria")
