"""
Moment expansion basis for fast synchrotron fitting.

Implements the SpyDust-style moment expansion:
SED(θ) ≈ SED(θ₀) + Σ (∂SED/∂θᵢ) Δθᵢ
"""

import numpy as np
from .models import jp_model, kp_model, ci_model, standard_model


class MomentBasis:
    """
    Moment expansion basis for synchrotron models.

    Precomputes SED and derivatives at pivot parameters,
    enabling fast linear fitting.
    """

    def __init__(self, frequencies, pivot_params, model_type='JP',
                 second_order=True):
        """
        Initialize moment basis.

        Parameters
        ----------
        frequencies : array
            Frequency array in Hz
        pivot_params : dict
            Pivot parameters: {'s': 2.3, 'log_nu_b': 9.5, ...}
        model_type : str
            'JP', 'KP', 'CI', 'CI-off'
        second_order : bool
            Whether to compute second derivatives
        """
        self.frequencies = np.atleast_1d(frequencies)
        self.model_type = model_type
        self.second_order = second_order

        # Set default pivot if not provided
        if pivot_params is None:
            pivot_params = {
                's': 2.3,
                'log_nu_b': 9.5,  # ν_b = 10^9.5 ≈ 30 GHz
            }
            if model_type == 'CI-off':
                pivot_params['T'] = 0.3

        self.pivot = pivot_params

        # Compute basis functions
        self._compute_basis()

    def _compute_basis(self):
        """
        Compute basis functions at pivot.

        Stores:
        - SED_0: SED at pivot
        - dS/ds: Derivative w.r.t. injection index
        - dS/dlog_nu_b: Derivative w.r.t. log break frequency
        - d2S/ds2, d2S/dlog_nu_b2, d2S/ds_dlog_nu: Second derivatives
        """
        freqs = self.frequencies
        s0 = self.pivot['s']
        log_nu0 = self.pivot['log_nu_b']
        nu0 = 10**log_nu0

        # Zeroth order: SED at pivot
        self.S_0 = standard_model(freqs, s0, nu0, self.model_type)

        # First derivatives (finite differences)
        ds = 0.1
        dlog_nu = 0.5

        # dS/ds
        S_plus = standard_model(freqs, s0 + ds, nu0, self.model_type)
        S_minus = standard_model(freqs, s0 - ds, nu0, self.model_type)
        self.dS_ds = (S_plus - S_minus) / (2 * ds)

        # dS/dlog_nu_b
        nu_plus = 10**(log_nu0 + dlog_nu)
        nu_minus = 10**(log_nu0 - dlog_nu)
        S_plus = standard_model(freqs, s0, nu_plus, self.model_type)
        S_minus = standard_model(freqs, s0, nu_minus, self.model_type)
        self.dS_dlog_nu = (S_plus - S_minus) / (2 * dlog_nu)

        # CI-off: also need dS/dT
        if self.model_type == 'CI-off' and 'T' in self.pivot:
            T0 = self.pivot['T']
            dT = 0.1

            S_plus = ci_model(freqs, s0, nu0, T0 + dT, active=False)
            S_minus = ci_model(freqs, s0, nu0, T0 - dT, active=False)
            self.dS_dT = (S_plus - S_minus) / (2 * dT)

        # Second derivatives (if requested)
        if self.second_order:
            # d²S/ds²
            self.d2S_ds2 = (standard_model(freqs, s0 + ds, nu0, self.model_type)
                          - 2*self.S_0
                          + standard_model(freqs, s0 - ds, nu0, self.model_type)) / ds**2

            # d²S/d(log_nu_b)²
            self.d2S_dlog_nu2 = (
                standard_model(freqs, s0, nu_plus, self.model_type)
                - 2*self.S_0
                + standard_model(freqs, s0, nu_minus, self.model_type)
            ) / dlog_nu**2

            # Mixed derivative
            self.d2S_ds_dlog_nu = self._compute_mixed_derivative(
                s0, log_nu0, ds, dlog_nu
            )

    def _compute_mixed_derivative(self, s0, log_nu0, ds, dlog_nu):
        """Compute mixed second derivative d²S/ds d(log_nu_b)"""
        freqs = self.frequencies

        # S(s+ds, log_nu+dlog_nu)
        S_pp = standard_model(freqs, s0 + ds, 10**(log_nu0 + dlog_nu), self.model_type)
        # S(s+ds, log_nu-dlog_nu)
        S_pm = standard_model(freqs, s0 + ds, 10**(log_nu0 - dlog_nu), self.model_type)
        # S(s-ds, log_nu+dlog_nu)
        S_mp = standard_model(freqs, s0 - ds, 10**(log_nu0 + dlog_nu), self.model_type)
        # S(s-ds, log_nu-dlog_nu)
        S_mm = standard_model(freqs, s0 - ds, 10**(log_nu0 - dlog_nu), self.model_type)

        return (S_pp - S_pm - S_mp + S_mm) / (4 * ds * dlog_nu)

    def evaluate(self, s, log_nu_b, T=None, amplitude=1.0,
                 use_second_order=False):
        """
        Evaluate SED using moment expansion.

        Parameters
        ----------
        s : float
            Injection index
        log_nu_b : float
            log10(break frequency)
        T : float, optional
            Remnant fraction (for CI-off)
        amplitude : float
            Overall normalization
        use_second_order : bool
            Whether to include second-order terms

        Returns
        -------
        spectrum : array
            SED at frequencies
        """
        delta_s = s - self.pivot['s']
        delta_log_nu = log_nu_b - self.pivot['log_nu_b']

        # First-order expansion
        sed = (self.S_0
               + delta_s * self.dS_ds
               + delta_log_nu * self.dS_dlog_nu)

        # Add T term for CI-off
        if T is not None and self.model_type == 'CI-off':
            delta_T = T - self.pivot['T']
            sed = sed + delta_T * self.dS_dT

        # Second-order terms
        if use_second_order and self.second_order:
            sed = (sed
                   + 0.5 * delta_s**2 * self.d2S_ds2
                   + 0.5 * delta_log_nu**2 * self.d2S_dlog_nu2
                   + delta_s * delta_log_nu * self.d2S_ds_dlog_nu)

        return amplitude * sed

    def get_design_matrix(self, include_second_order=False):
        """
        Get design matrix for linear fitting.

        Returns matrix M where d = M·θ gives the data.
        Parameters θ = [A, Δs, Δlog_nu_b, ...]

        Returns
        -------
        M : array [n_freq, n_params]
            Design matrix
        """
        n_freq = len(self.frequencies)
        n_params = 3  # A, Δs, Δlog_nu

        if self.model_type == 'CI-off':
            n_params += 1  # +ΔT

        M = np.zeros((n_freq, n_params))

        # Column 0: Amplitude (just S_0)
        M[:, 0] = self.S_0

        # Column 1: Δs coefficient
        M[:, 1] = self.dS_ds

        # Column 2: Δlog_nu coefficient
        M[:, 2] = self.dS_dlog_nu

        # Column 3: ΔT (for CI-off)
        if self.model_type == 'CI-off':
            M[:, 3] = self.dS_dT

        return M

    def linear_predict(self, params):
        """
        Fast linear prediction from parameters.

        Parameters
        ----------
        params : array [n_params]
            [amplitude, delta_s, delta_log_nu, (delta_T)]

        Returns
        -------
        spectrum : array
            Predicted SED
        """
        M = self.get_design_matrix()
        return M @ params


def compute_moment_basis(frequencies, pivot_params=None, model_type='JP'):
    """
    Convenience function to create moment basis.

    Parameters
    ----------
    frequencies : array
        Frequencies in Hz
    pivot_params : dict or None
        Pivot parameters. If None, uses defaults.
    model_type : str
        'JP', 'KP', 'CI', 'CI-off'

    Returns
    -------
    basis : MomentBasis
        Precomputed moment expansion basis
    """
    return MomentBasis(frequencies, pivot_params, model_type)
