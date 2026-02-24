#!/usr/bin/env python
"""
Basic example: Fit a synchrotron spectrum.

Demonstrates moment expansion fitting to a single spectrum.
"""

import numpy as np
import matplotlib.pyplot as plt
from localsync import MomentBasis, fit_spectrum


def main():
    print("SpySynch Basic Fitting Example")
    print("=" * 50)

    # Generate synthetic data
    print("\n1. Generating synthetic data...")

    # True parameters
    s_true = 2.3
    log_nu_b_true = 9.5  # ν_b = 10^9.5 ≈ 30 GHz
    amplitude_true = 1.0

    # Frequencies (GHz)
    frequencies = np.array([23, 30, 44, 70, 100, 143, 217, 353]) * 1e9  # Hz

    # Generate JP model spectrum
    from localsync.models import jp_model
    spectrum_true = amplitude_true * jp_model(frequencies, s_true, 10**log_nu_b_true)

    # Add noise
    noise_level = 0.05 * spectrum_true  # 5% noise
    data = spectrum_true + np.random.randn(len(frequencies)) * noise_level

    print(f"   True s = {s_true}")
    print(f"   True log ν_b = {log_nu_b_true}")
    print(f"   S/N ratio ≈ {np.mean(spectrum_true / noise_level):.1f}")

    # Fit with moment expansion
    print("\n2. Fitting with moment expansion...")

    result = fit_spectrum(
        data, frequencies, model_type='JP',
        data_error=noise_level, method='linear'
    )

    print(f"   Fitted s = {result['s']:.3f} ± {result['uncertainties'][1]:.3f}")
    print(f"   Fitted log ν_b = {result['log_nu_b']:.3f} ± {result['uncertainties'][2]:.3f}")
    print(f"   χ²/dof = {result['chi2_reduced']:.2f}")

    # Compare with true values
    print("\n3. Comparison:")
    print(f"   s: true={s_true:.3f}, fitted={result['s']:.3f}, "
          f"error={abs(result['s'] - s_true):.3f}")
    print(f"   log ν_b: true={log_nu_b_true:.3f}, fitted={result['log_nu_b']:.3f}, "
          f"error={abs(result['log_nu_b'] - log_nu_b_true):.3f}")

    # Plot
    print("\n4. Creating plot...")
    plt.figure(figsize=(10, 6))

    # Data
    plt.errorbar(frequencies / 1e9, data, yerr=noise_level,
                fmt='o', label='Data', capsize=3)

    # True model
    plt.plot(frequencies / 1e9, spectrum_true, 'k--',
            label=f'True model (s={s_true})')

    # Fitted model
    fitted_spectrum = (result['amplitude'] *
                      jp_model(frequencies, result['s'], result['nu_b']))
    plt.plot(frequencies / 1e9, fitted_spectrum, 'r-',
            label=f'Fitted model (s={result["s"]:.2f})')

    plt.xlabel('Frequency (GHz)')
    plt.ylabel('Intensity (arb. units)')
    plt.title('SpySynch Moment Expansion Fitting')
    plt.legend()
    plt.xscale('log')
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig('example_fit.png', dpi=150)
    print("   Saved to example_fit.png")

    print("\nDone!")


if __name__ == '__main__':
    main()
