"""
LocalSync: Local Bubble Synchrotron Emission Model

A package for modeling Galactic synchrotron emission using physical
synchrotron models (JP, KP, CI) with moment expansion for fast fitting.

Based on the SpyDust formalism applied to synchrotron emission.


"""

__version__ = "0.1.0"
__author__ = "LocalSync Contributors"

from .models import jp_model, kp_model, ci_model, besselK53, standard_model
from .basis import compute_moment_basis, MomentBasis
from .fitting import fit_spectrum, LinearFitter
from .utils import spectral_units, validate_parameters, spectral_age
from .tomography import LocalBubbleTomography
from .lightbox import LightBoxFitter, SphericalForeground, CylindricalBackground

__all__ = [
    'jp_model',
    'kp_model',
    'ci_model',
    'besselK53',
    'standard_model',
    'compute_moment_basis',
    'MomentBasis',
    'fit_spectrum',
    'LinearFitter',
    'spectral_units',
    'validate_parameters',
    'spectral_age',
    'LocalBubbleTomography',
    'LightBoxFitter',
    'SphericalForeground',
    'CylindricalBackground',
]
