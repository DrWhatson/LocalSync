"""
Fast parameter fitting using moment expansion.

Implements linear least squares and iterative refinement.
"""

import numpy as np
from scipy.optimize import minimize, least_squares
from .basis import MomentBasis


class LinearFitter:
    """
    Fast linear fitting using moment expansion basis.

    Fits parameters by solving: min ||d - M·θ||²
    where M is the moment expansion design matrix.
    """

    def __init__(self, basis):
        """
        Initialize fitter with moment basis.

        Parameters
        ----------
        basis : MomentBasis
            Precomputed moment expansion basis
        """
        self.basis = basis
        self.M = basis.get_design_matrix()
        self.n_params = self.M.shape[1]

    def fit(self, data, data_error=None, method='lsqr'):
        """
        Fit parameters to data.

        Parameters
        ----------
        data : array
            Observed SED
        data_error : array or None
            Uncertainties on data. If None, assumes uniform.
        method : str
            'lsqr' (default), 'lstsq', or 'ridge'

        Returns
        -------
        params : dict
            Fitted parameters with uncertainties
        """
        data = np.atleast_1d(data)

        # Weight by error if provided
        if data_error is not None:
            weights = 1.0 / data_error
            M_weighted = self.M * weights[:, np.newaxis]
            d_weighted = data * weights
        else:
            M_weighted = self.M
            d_weighted = data

        # Solve linear system
        if method == 'lsqr':
            from scipy.sparse.linalg import lsqr
            result, istop = lsqr(M_weighted, d_weighted)[:2]
            params_array = result

        elif method == 'lstsq':
            params_array, residuals, rank, s = np.linalg.lstsq(
                M_weighted, d_weighted, rcond=None
            )

        elif method == 'ridge':
            # Ridge regression with small regularization
            alpha = 0.01
            MtM = M_weighted.T @ M_weighted
            MtM += alpha * np.eye(self.n_params)
            params_array = np.linalg.solve(MtM, M_weighted.T @ d_weighted)

        else:
            raise ValueError(f"Unknown method: {method}")

        # Extract physical parameters
        params = self._array_to_dict(params_array)

        # Estimate uncertainties from covariance
        try:
            covariance = self._compute_covariance(M_weighted, data_error)
            params['covariance'] = covariance
            params['uncertainties'] = np.sqrt(np.diag(covariance))
        except:
            params['uncertainties'] = None

        # Compute goodness of fit
        model = self.basis.linear_predict(params_array)
        chi2 = np.sum(((data - model) / data_error)**2) if data_error is not None else np.sum((data - model)**2)
        dof = len(data) - self.n_params

        params['chi2'] = chi2
        params['dof'] = dof
        params['chi2_reduced'] = chi2 / dof if dof > 0 else np.inf
        params['residuals'] = data - model

        return params

    def _array_to_dict(self, params_array):
        """Convert parameter array to dictionary."""
        params = {
            'amplitude': params_array[0],
            'delta_s': params_array[1],
            'delta_log_nu_b': params_array[2],
        }

        # Compute absolute parameters
        params['s'] = self.basis.pivot['s'] + params['delta_s']
        params['log_nu_b'] = self.basis.pivot['log_nu_b'] + params['delta_log_nu_b']
        params['nu_b'] = 10**params['log_nu_b']

        if self.basis.model_type == 'CI-off' and len(params_array) > 3:
            params['delta_T'] = params_array[3]
            params['T'] = self.basis.pivot['T'] + params['delta_T']

        return params

    def _compute_covariance(self, M, data_error):
        """Compute parameter covariance matrix."""
        if data_error is None:
            # Assume unit errors
            data_error = np.ones(M.shape[0])

        MtM = M.T @ (M / data_error[:, np.newaxis]**2)
        covariance = np.linalg.inv(MtM)
        return covariance


class NonLinearFitter:
    """
    Non-linear fitting using full synchrotron models.

    Uses moment expansion for initialization, then refines
    with full model calculation.
    """

    def __init__(self, frequencies, model_type='JP'):
        """
        Initialize non-linear fitter.

        Parameters
        ----------
        frequencies : array
            Frequencies in Hz
        model_type : str
            'JP', 'KP', 'CI', 'CI-off'
        """
        self.frequencies = frequencies
        self.model_type = model_type

        # Create basis for initialization
        self.basis = MomentBasis(frequencies, None, model_type)
        self.linear_fitter = LinearFitter(self.basis)

    def fit(self, data, data_error=None, method='moment_then_full'):
        """
        Fit parameters using full model.

        Parameters
        ----------
        data : array
            Observed SED
        data_error : array or None
            Uncertainties
        method : str
            'moment': Linear fit only (fast)
            'full': Non-linear optimization (slow, accurate)
            'moment_then_full': Both (recommended)

        Returns
        -------
        params : dict
            Best-fit parameters
        """
        if method in ['moment', 'moment_then_full']:
            # Step 1: Fast linear fit for initialization
            print("Running moment expansion fit...")
            linear_params = self.linear_fitter.fit(data, data_error)
            initial_guess = [
                linear_params['s'],
                linear_params['log_nu_b'],
            ]
            if self.model_type == 'CI-off':
                initial_guess.append(linear_params.get('T', 0.3))
        else:
            # Use defaults
            initial_guess = [2.3, 9.5]
            if self.model_type == 'CI-off':
                initial_guess.append(0.3)

        if method == 'moment':
            return linear_params

        # Step 2: Non-linear refinement (if requested)
        print("Running non-linear refinement...")

        from .models import standard_model

        def chi2(params):
            s, log_nu_b = params[0], params[1]
            nu_b = 10**log_nu_b

            if self.model_type == 'CI-off':
                T = params[2]
                model = standard_model(self.frequencies, s, nu_b, 'CI-off')
            else:
                model = standard_model(self.frequencies, s, nu_b, self.model_type)

            if data_error is not None:
                return np.sum(((data - model) / data_error)**2)
            else:
                return np.sum((data - model)**2)

        # Bounds for physical parameters
        bounds = ([2.0, 8.0], [3.0, 11.0])
        if self.model_type == 'CI-off':
            bounds = ([2.0, 8.0, 0.0], [3.0, 11.0, 1.0])

        result = minimize(
            chi2,
            initial_guess,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 1000}
        )

        # Package results
        params = {
            's': result.x[0],
            'log_nu_b': result.x[1],
            'nu_b': 10**result.x[1],
            'chi2': result.fun,
            'success': result.success,
            'nfev': result.nfev,
        }

        if self.model_type == 'CI-off':
            params['T'] = result.x[2]

        # Compute final residuals
        model = standard_model(self.frequencies, params['s'],
                              params['nu_b'], self.model_type)
        params['residuals'] = data - model

        return params


def fit_spectrum(data, frequencies, model_type='JP', data_error=None,
               method='linear', pivot_params=None):
    """
    Convenience function to fit synchrotron spectrum.

    Parameters
    ----------
    data : array
        Observed SED
    frequencies : array
        Frequencies in Hz
    model_type : str
        'JP', 'KP', 'CI', 'CI-off'
    data_error : array or None
        Uncertainties
    method : str
        'linear' (fast), 'nonlinear' (accurate), 'both' (recommended)
    pivot_params : dict or None
        Pivot parameters for moment expansion

    Returns
    -------
    result : dict
        Fit results with parameters and statistics
    """
    if method == 'linear':
        basis = MomentBasis(frequencies, pivot_params, model_type)
        fitter = LinearFitter(basis)
        return fitter.fit(data, data_error)

    elif method == 'nonlinear':
        fitter = NonLinearFitter(frequencies, model_type)
        return fitter.fit(data, data_error, method='full')

    elif method == 'both':
        fitter = NonLinearFitter(frequencies, model_type)
        return fitter.fit(data, data_error, method='moment_then_full')

    else:
        raise ValueError(f"Unknown method: {method}")


class SpatialFitter:
    """
    Fit spatial field of parameters (e.g., entire sky).

    Uses moment expansion for fast fitting of many pixels.
    """

    def __init__(self, frequencies, nside, model_type='JP'):
        """
        Initialize spatial fitter.

        Parameters
        ----------
        frequencies : array
            Frequencies in Hz
        nside : int
            HEALPix resolution
        model_type : str
            Synchrotron model type
        """
        self.frequencies = frequencies
        self.nside = nside
        self.npix = 12 * nside**2
        self.model_type = model_type

        # Precompute basis
        self.basis = MomentBasis(frequencies, None, model_type)
        self.linear_fitter = LinearFitter(self.basis)

    def fit_cube(self, data_cube, data_error_cube=None, mask=None,
                verbose=True):
        """
        Fit parameters for entire data cube.

        Parameters
        ----------
        data_cube : array [n_freq, npix]
            Data for all pixels
        data_error_cube : array [n_freq, npix] or None
            Uncertainties
        mask : array [npix] or None
            Mask of valid pixels to fit
        verbose : bool
            Print progress

        Returns
        -------
        params_cube : dict
            Parameter maps: 's', 'log_nu_b', etc.
        """
        if mask is None:
            mask = np.ones(self.npix, dtype=bool)

        # Initialize output arrays
        s_map = np.full(self.npix, np.nan)
        log_nu_b_map = np.full(self.npix, np.nan)
        chi2_map = np.full(self.npix, np.nan)

        # Fit each pixel
        valid_pixels = np.where(mask)[0]
        n_valid = len(valid_pixels)

        for i, pix in enumerate(valid_pixels):
            if verbose and i % 1000 == 0:
                print(f"Fitting pixel {i}/{n_valid}...")

            data = data_cube[:, pix]

            if data_error_cube is not None:
                data_error = data_error_cube[:, pix]
            else:
                data_error = None

            # Skip if no data
            if np.all(~np.isfinite(data)) or np.all(data == 0):
                continue

            try:
                params = self.linear_fitter.fit(data, data_error)
                s_map[pix] = params['s']
                log_nu_b_map[pix] = params['log_nu_b']
                chi2_map[pix] = params['chi2_reduced']
            except:
                # Fit failed - leave as NaN
                pass

        return {
            's': s_map,
            'log_nu_b': log_nu_b_map,
            'nu_b': 10**log_nu_b_map,
            'chi2_reduced': chi2_map,
        }
