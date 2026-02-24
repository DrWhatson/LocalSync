#!/usr/bin/env python
"""
Test SpySynch installation.

Run this to verify everything is installed correctly.
"""

import sys


def test_imports():
    """Test that all modules import correctly."""
    print("Testing imports...")

    try:
        import localsync
        print("  ✓ localsync imports successfully")

        from localsync import jp_model, MomentBasis, fit_spectrum, LocalBubbleTomography
        print("  ✓ All main classes available")

        return True
    except ImportError as e:
        print(f"  ✗ Import failed: {e}")
        return False


def test_physics():
    """Test physics models."""
    print("\nTesting physics models...")

    import numpy as np
    from localsync import jp_model, MomentBasis

    try:
        # Test JP model
        freqs = np.linspace(1e9, 100e9, 100)
        spectrum = jp_model(freqs, injection_index=2.3, break_frequency=10e9)
        print(f"  ✓ JP model works (shape: {spectrum.shape})")

        # Test moment basis
        basis = MomentBasis(freqs, {'s': 2.3, 'log_nu_b': 9.5})
        print(f"  ✓ Moment basis created (n_freq: {len(basis.S_0)})")

        # Test evaluation
        sed = basis.evaluate(s=2.5, log_nu_b=10.0)
        print(f"  ✓ Moment expansion works")

        return True
    except Exception as e:
        print(f"  ✗ Physics test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_fitting():
    """Test fitting routines."""
    print("\nTesting fitting...")

    import numpy as np
    from localsync import fit_spectrum, jp_model

    try:
        # Generate test data
        freqs = np.array([23, 30, 44, 70, 100]) * 1e9
        true_spectrum = jp_model(freqs, injection_index=2.3, break_frequency=30e9)
        data = true_spectrum * (1 + 0.1 * np.random.randn(len(freqs)))

        # Fit
        result = fit_spectrum(data, freqs, model_type='JP', method='linear')
        print(f"  ✓ Linear fit works")
        print(f"    Fitted s = {result['s']:.2f}")
        print(f"    Reduced χ² = {result['chi2_reduced']:.2f}")

        return True
    except Exception as e:
        print(f"  ✗ Fitting test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_tomography():
    """Test tomography module."""
    print("\nTesting tomography...")

    import numpy as np

    try:
        from localsync import LocalBubbleTomography

        # Create tomography object
        tomography = LocalBubbleTomography(nside=32, r_bins=np.linspace(0, 200, 5))
        print(f"  ✓ Tomography object created")
        print(f"    n_voxels: {tomography.n_voxels}")

        return True
    except Exception as e:
        print(f"  ✗ Tomography test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("=" * 60)
    print("SpySynch Installation Test")
    print("=" * 60)

    tests = [
        ("Imports", test_imports),
        ("Physics", test_physics),
        ("Fitting", test_fitting),
        ("Tomography", test_tomography),
    ]

    results = []
    for name, test_func in tests:
        try:
            results.append(test_func())
        except Exception as e:
            print(f"\n✗ {name} failed: {e}")
            results.append(False)

    print("\n" + "=" * 60)
    if all(results):
        print("All tests passed! SpySynch is ready to use.")
        print("\nNext steps:")
        print("  1. Run: python examples/example_basic_fit.py")
        print("  2. Run: python examples/example_tomography.py")
        print("  3. Integrate with your WP3.2 data")
        return 0
    else:
        print("Some tests failed. Please check the errors above.")
        return 1
    print("=" * 60)


if __name__ == '__main__':
    sys.exit(main())
