"""
Utility functions for SpySynch.
"""

import numpy as np
import healpy as hp
from scipy.special import kv


def besselK53(x):
    """
    Modified Bessel function K_{5/3}(x).
    Used in synchrotron emissivity calculation.
    """
    return kv(5/3, x)


def spectral_units(data, input_unit, output_unit='Hz'):
    """
    Convert frequency/spectral data between units.

    Parameters
    ----------
    data : array
        Input data values
    input_unit : str
        'Hz', 'MHz', 'GHz', 'kHz'
    output_unit : str
        Target unit

    Returns
    -------
    converted : array
        Data in output units
    """
    # Frequency conversion factors
    factors = {
        'Hz': 1.0,
        'kHz': 1e3,
        'MHz': 1e6,
        'GHz': 1e9,
        'THz': 1e12,
    }

    if input_unit not in factors:
        raise ValueError(f"Unknown input unit: {input_unit}")
    if output_unit not in factors:
        raise ValueError(f"Unknown output unit: {output_unit}")

    return data * factors[input_unit] / factors[output_unit]


def validate_parameters(s, log_nu_b, model_type='JP', T=None):
    """
    Validate physical parameter ranges.

    Parameters
    ----------
    s : float
        Injection index
    log_nu_b : float
        log10(break frequency)
    model_type : str
        Model type
    T : float or None
        Remnant fraction (for CI-off)

    Returns
    -------
    is_valid : bool
        Whether parameters are physical
    message : str
        Error message if invalid
    """
    if s < 1.5 or s > 4.0:
        return False, f"Injection index {s} out of range [1.5, 4.0]"

    if log_nu_b < 6 or log_nu_b > 13:
        return False, f"log break frequency {log_nu_b} out of range [6, 13]"

    if model_type == 'CI-off' and T is not None:
        if T < 0 or T > 1:
            return False, f"Remnant fraction {T} out of range [0, 1]"

    return True, "Valid"


def spectral_age(nu_b, B_field, model_type='JP'):
    """
    Estimate spectral age from break frequency and B-field.

    τ_syn ∝ B^(-3/2) / (B² + B_IC²) / sqrt(ν_b)

    Parameters
    ----------
    nu_b : float
        Break frequency in Hz
    B_field : float
        Magnetic field strength in Tesla
    model_type : str
        Model type (affects age calculation)

    Returns
    -------
    age : float
        Spectral age in Myr
    """
    # Physical constants
    c = 2.998e8  # m/s
    m_e = 9.109e-31  # kg
    e = 1.602e-19  # C

    # Critical B-field for IC losses
    B_ic = 3.25e-9  # Tesla (CMB equivalent at z=0)

    # Break frequency relation
    # ν_b ∝ B (E_crit)² where E_crit is critical energy
    # E_crit decreases with time due to cooling

    if model_type == 'JP':
        const = 1.0
    elif model_type == 'KP':
        const = 1.0  # Similar to JP
    else:  # CI
        const = 1.5  # Older apparent age

    # Synchrotron age formula (Turner 2018)
    tau_syn = const * 159.0 * (B_field**0.5 / (B_field**2 + B_ic**2))
    tau_syn /= np.sqrt(nu_b / 1e9)  # Normalize to 1 GHz

    return tau_syn  # Myr


def compute_chi2(data, model, error=None, dof=None):
    """
    Compute chi-squared statistic.

    Parameters
    ----------
    data : array
        Observed data
    model : array
        Model prediction
    error : array or None
        Measurement uncertainties
    dof : int or None
        Degrees of freedom

    Returns
    -------
    chi2 : dict
        Chi2 statistics
    """
    if error is not None:
        residuals = (data - model) / error
    else:
        residuals = data - model
        error = np.ones_like(data)

    chi2 = np.sum(residuals**2)
    n_data = len(data)

    if dof is None:
        dof = n_data

    return {
        'chi2': chi2,
        'dof': dof,
        'chi2_reduced': chi2 / dof if dof > 0 else np.inf,
        'residuals': data - model,
        'rms': np.std(data - model),
        'rms_relative': np.std((data - model) / data) if np.any(data != 0) else np.inf,
    }


def mask_regions(nside, region_type, **kwargs):
    """
    Create HEALPix masks for common regions.

    Parameters
    ----------
    nside : int
        HEALPix resolution
    region_type : str
        'local_bubble', 'galactic_plane', 'north_polar_spur', etc.
    **kwargs : dict
        Region-specific parameters

    Returns
    -------
    mask : array
        Boolean mask (True = in region)
    """
    npix = hp.nside2npix(nside)
    mask = np.zeros(npix, dtype=bool)

    if region_type == 'local_bubble':
        # Approximate: high latitude sphere around (0,0)
        l, b = hp.pix2ang(nside, np.arange(npix), lonlat=True)
        radius = kwargs.get('radius', 60)  # degrees
        mask = (np.abs(b) < radius) & (np.abs(l) < 90)

    elif region_type == 'galactic_plane':
        l, b = hp.pix2ang(nside, np.arange(npix), lonlat=True)
        width = kwargs.get('width', 10)  # degrees
        mask = np.abs(b) < width

    elif region_type == 'north_polar_spur':
        l, b = hp.pix2ang(nside, np.arange(npix), lonlat=True)
        mask = ((l > 10) & (l < 50) &
                (b > 10) & (b < 50))

    elif region_type == 'high_latitude':
        l, b = hp.pix2ang(nside, np.arange(npix), lonlat=True)
        cutoff = kwargs.get('cutoff', 30)  # degrees
        mask = np.abs(b) > cutoff

    return mask


def smooth_healpix(map_in, fwhm_deg, nside=None):
    """
    Smooth HEALPix map with Gaussian beam.

    Parameters
    ----------
    map_in : array
        Input HEALPix map
    fwhm_deg : float
        Full width at half maximum in degrees
    nside : int or None
        If different from input

    Returns
    -------
    smoothed : array
        Smoothed map
    """
    if nside is not None and nside != hp.get_nside(map_in):
        map_in = hp.ud_grade(map_in, nside_out=nside)

    fwhm_rad = np.radians(fwhm_deg)
    return hp.smoothing(map_in, fwhm=fwhm_rad)


class ProgressBar:
    """Simple progress bar for iterations."""

    def __init__(self, total, desc="Processing"):
        self.total = total
        self.desc = desc
        self.current = 0

    def update(self, n=1):
        self.current += n
        percent = 100 * self.current / self.total
        print(f"\r{self.desc}: {percent:.1f}% ({self.current}/{self.total})",
              end='', flush=True)
        if self.current >= self.total:
            print()

    def close(self):
        print()
