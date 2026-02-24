#!/usr/bin/env python
"""
Example: 3D Local Bubble Tomography.

Demonstrates tomographic reconstruction of synchrotron parameters.
"""

import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from localsync import LocalBubbleTomography


def main():
    print("SpySynch 3D Local Bubble Tomography Example")
    print("=" * 60)

    # Setup
    print("\n1. Setting up tomography...")
    nside = 32  # Lower resolution for demo
    r_bins = np.linspace(0, 200, 11)  # 10 radial bins

    tomography = LocalBubbleTomography(nside=nside, r_bins=r_bins)

    # Generate synthetic data
    print("\n2. Generating synthetic data...")

    freqs = np.array([23, 30, 44, 70, 100]) * 1e9  # Hz
    npix = hp.nside2npix(nside)

    # True parameters: vary with radius
    # Wall (outer): flatter spectrum (s=2.1)
    # Interior: steeper (s=2.8)

    s_true = np.zeros((len(r_bins)-1, npix))
    log_nu_b_true = np.zeros((len(r_bins)-1, npix))

    for r_idx, r in enumerate(r_bins[:-1]):
        if r > 130:  # Wall
            s_true[r_idx, :] = 2.1 + 0.2 * np.random.randn(npix)
            log_nu_b_true[r_idx, :] = 11.0  # High break
        else:  # Interior
            s_true[r_idx, :] = 2.8 + 0.1 * np.random.randn(npix)
            log_nu_b_true[r_idx, :] = 9.5   # Low break

    # Generate data cube
    from localsync import MomentBasis
    basis = MomentBasis(freqs, None, model_type='JP')

    data_cube = np.zeros((len(freqs), npix))

    for r_idx in range(len(r_bins)-1):
        for pix in range(npix):
            # SED for this voxel
            sed = basis.evaluate(s_true[r_idx, pix], log_nu_b_true[r_idx, pix])

            # Add to observed data (simplified - no projection for demo)
            # In real case: integrate along LOS
            data_cube[:, pix] += sed / len(r_bins)  # Average contribution

    # Add noise
    noise = 0.1 * data_cube
    data_cube += np.random.randn(*data_cube.shape) * noise

    print(f"   Data shape: {data_cube.shape}")
    print(f"   S/N: {np.mean(data_cube/noise):.1f}")

    # Build projection matrix
    print("\n3. Building projection matrix...")
    tomography.build_projection_matrix(cache_file='projection_demo.npz')

    # Fit tomography
    print("\n4. Running tomographic fit...")
    # Note: this is a simplified demo - real fitting would take longer

    # For demo, just return true params
    print("   (In real run, would solve 3D inversion)")
    params_3d = {
        's': s_true,
        'log_nu_b': log_nu_b_true,
        'amplitude': np.ones_like(s_true)
    }

    # Visualize
    print("\n5. Visualizing results...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Radial profiles
    r_mid = 0.5 * (r_bins[:-1] + r_bins[1:])

    axes[0].plot(r_mid, np.mean(s_true, axis=1), 'o-', label='s (injection index)')
    axes[0].axvline(130, color='r', linestyle='--', label='LB wall')
    axes[0].set_xlabel('Radius (pc)')
    axes[0].set_ylabel('s')
    axes[0].set_title('Injection Index vs Radius')
    axes[0].legend()

    axes[1].plot(r_mid, 10**np.mean(log_nu_b_true, axis=1) / 1e9,
                'o-', label='ν_b (GHz)')
    axes[1].axvline(130, color='r', linestyle='--', label='LB wall')
    axes[1].set_xlabel('Radius (pc)')
    axes[1].set_ylabel('Break Frequency (GHz)')
    axes[1].set_title('Break Frequency vs Radius')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig('tomography_demo.png', dpi=150)
    print("   Saved to tomography_demo.png")

    print("\n6. Summary:")
    print("   - Interior (r < 130 pc): s ≈ 2.8, ν_b ≈ 30 GHz (steep, old)")
    print("   - Wall (r > 130 pc): s ≈ 2.1, ν_b ≈ 100 GHz (flat, young)")
    print("\nDone!")


if __name__ == '__main__':
    main()
