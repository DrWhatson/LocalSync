"""
Synchrotron spectrum models: JP, KP, CI

Implementation of standard synchrotron models from Turner & Quici 2018.
"""

import numpy as np
from scipy.special import kv


def besselK53(x):
    """
    Modified Bessel function K_{5/3}(x).
    Used in synchrotron emissivity calculation.
    """
    return kv(5/3, x)


def synchrotron_emissivity(x):
    """
    Synchrotron emissivity function F(x).

    x = ν/ν_c where ν_c is critical frequency
    F(x) = x ∫_x^∞ K_{5/3}(t) dt
    """
    # Approximation for small/large x
    if x < 1e-10:
        return 2.15 * x**(1/3)
    elif x > 100:
        return np.sqrt(np.pi/2) * np.exp(-x) * np.sqrt(x)
    else:
        # Full integral
        from scipy import integrate
        result, _ = integrate.quad(lambda t: kv(5/3, t), x, np.inf)
        return x * result


def jp_model(frequencies, injection_index, break_frequency, pitch_angle_avg=True):
    """
    Jaffe-Perola (JP) synchrotron model.

    Impulsive injection with isotropic pitch angle distribution.
    Includes electron cooling and pitch angle scattering.

    Parameters
    ----------
    frequencies : array
        Frequencies in Hz
    injection_index : float
        Energy injection index s (typically 2.0-3.0)
    break_frequency : float
        Break frequency ν_b in Hz (where cooling becomes significant)
    pitch_angle_avg : bool
        Whether to average over pitch angles

    Returns
    -------
    spectrum : array
        SED(frequencies) normalized
    """
    frequencies = np.atleast_1d(frequencies)
    s = injection_index
    nu_b = break_frequency

    # Dimensionless frequency ratio
    x = frequencies / nu_b

    # JP spectrum calculation
    if pitch_angle_avg:
        # Average over pitch angles (more accurate)
        n_alpha = 32
        # Start from small angle to avoid sin(alpha)=0
        alpha_vals = np.linspace(0.01, np.pi/2, n_alpha)
        d_alpha = (np.pi/2 - 0.01) / n_alpha

        spectrum = np.zeros_like(frequencies)

        for alpha in alpha_vals:
            sin_alpha = np.sin(alpha)
            if sin_alpha < 1e-10:
                continue  # Skip very small angles
            x_crit = x / sin_alpha
            x_crit = np.clip(x_crit, 0, 1e10)  # Prevent overflow
            gamma = (s - 1) / 2

            spec_alpha = np.zeros_like(frequencies)

            below_break = x_crit < 1
            above_break = ~below_break

            if np.any(below_break):
                spec_alpha[below_break] = (frequencies[below_break] / nu_b) ** (-gamma)

            if np.any(above_break):
                spec_alpha[above_break] = (
                    (frequencies[above_break] / nu_b) ** (-gamma - 0.5) *
                    (np.sqrt(x_crit[above_break]) - 1)**(s - 2)
                )

            spectrum += spec_alpha * sin_alpha**(s/2 + 1) * d_alpha
    else:
        # Simplified: no pitch angle averaging
        gamma = (s - 1) / 2

        below_break = x < 1
        above_break = ~below_break

        spectrum = np.zeros_like(frequencies)

        if np.any(below_break):
            spectrum[below_break] = x[below_break] ** (-gamma)

        if np.any(above_break):
            spectrum[above_break] = x[above_break] ** (-gamma - 0.5)

    return spectrum


def kp_model(frequencies, injection_index, break_frequency):
    """
    Kardashev-Pacholczyk (KP) synchrotron model.

    Impulsive injection with constant pitch angle distribution.
    No pitch angle scattering (unlike JP).

    Parameters
    ----------
    frequencies : array
        Frequencies in Hz
    injection_index : float
        Energy injection index s
    break_frequency : float
        Break frequency ν_b in Hz

    Returns
    -------
    spectrum : array
        SED(frequencies) normalized
    """
    frequencies = np.atleast_1d(frequencies)
    s = injection_index
    nu_b = break_frequency

    x = frequencies / nu_b
    gamma = (s - 1) / 2

    below_break = x < 1
    above_break = ~below_break

    spectrum = np.zeros_like(frequencies)

    if np.any(below_break):
        spectrum[below_break] = x[below_break] ** (-gamma)

    if np.any(above_break):
        spectrum[above_break] = (
            x[above_break] ** (-gamma - 1) *
            (np.sqrt(x[above_break]) - 1)**(s - 1)
        )

    return spectrum


def ci_model(frequencies, injection_index, break_frequency,
             remnant_fraction=0.0, active=True):
    """
    Continuous Injection (CI) synchrotron model.

    Continuous injection of electrons from t=0 to τ (source age).

    Parameters
    ----------
    frequencies : array
        Frequencies in Hz
    injection_index : float
        Energy injection index s
    break_frequency : float
        Break frequency ν_b in Hz (corresponds to cooling time)
    remnant_fraction : float
        T = t_off/τ, fraction of time source was inactive [0,1]
        If T=0: CI-on (still active)
        If T>0: CI-off (had inactive period)
    active : bool
        If True, CI-on model; if False, CI-off

    Returns
    -------
    spectrum : array
        SED(frequencies) normalized
    """
    frequencies = np.atleast_1d(frequencies)
    s = injection_index
    nu_b = break_frequency
    T = remnant_fraction

    x = frequencies / nu_b
    gamma = (s - 1) / 2

    if active or T == 0:
        # CI-on: still injecting
        below_break = x < 1
        intermediate = (x >= 1) & (x < 10)
        above_break = x >= 10

        spectrum = np.zeros_like(frequencies)

        if np.any(below_break):
            spectrum[below_break] = x[below_break] ** (-gamma)

        if np.any(intermediate):
            spectrum[intermediate] = (
                x[intermediate] ** (-gamma) *
                (1 - x[intermediate]**(-0.5))
            )

        if np.any(above_break):
            spectrum[above_break] = x[above_break] ** (-gamma - 0.5)
    else:
        # CI-off: had inactive period
        x_star = x * T**2

        below_star = x < x_star
        between = (x >= x_star) & (x < 1)
        above_break = x >= 1

        spectrum = np.zeros_like(frequencies)

        if np.any(below_star):
            spectrum[below_star] = x[below_star] ** (-gamma)

        if np.any(between):
            spectrum[between] = (
                x[between]**(-gamma) *
                (1 - (np.sqrt(x[between]) - np.sqrt(x_star[between])))**(s-1)
            )

        if np.any(above_break):
            spectrum[above_break] = x[above_break] ** (-gamma - 0.5)

    return spectrum


def standard_model(frequencies, injection_index, break_frequency,
                   model_type='JP'):
    """
    Convenience function to call different standard models.

    Parameters
    ----------
    frequencies : array
        Frequencies in Hz
    injection_index : float
        Energy injection index s
    break_frequency : float
        Break frequency ν_b
    model_type : str
        'JP', 'KP', 'CI', or 'CI-off'

    Returns
    -------
    spectrum : array
        Model spectrum
    """
    if model_type == 'JP':
        return jp_model(frequencies, injection_index, break_frequency)
    elif model_type == 'KP':
        return kp_model(frequencies, injection_index, break_frequency)
    elif model_type == 'CI':
        return ci_model(frequencies, injection_index, break_frequency,
                       active=True)
    elif model_type == 'CI-off':
        return ci_model(frequencies, injection_index, break_frequency,
                       active=False)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
