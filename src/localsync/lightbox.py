"""
Light Box Iterative Fitter

Iterative fitting scheme for:
- Local Bubble: 2D spherical foreground screen (0-200 pc)
- Galactic Plane: 3D cylindrical background (>200 pc)

The approach:
1. Fix Galaxy, fit LB to residuals
2. Fix LB, fit Galaxy to residuals
3. Iterate until convergence
"""

import numpy as np
import healpy as hp
from scipy.optimize import minimize
import logging

from .basis import MomentBasis
from .fitting import LinearFitter
from .models import jp_model, ci_model

logger = logging.getLogger(__name__)


class SphericalForeground:
    """
    2D spherical foreground screen (Local Bubble).

    Parameters are defined on the sphere (HEALPix grid).
    """

    def __init__(self, nside=128, frequencies=None, pivot_params=None):
        """
        Parameters
        ----------
        nside : int
            HEALPix resolution
        frequencies : array
            Frequencies in Hz
        pivot_params : dict
            Pivot parameters for moment expansion
        """
        self.nside = nside
        self.npix = hp.nside2npix(nside)
        self.frequencies = np.atleast_1d(frequencies) if frequencies is not None else None

        # Default pivot parameters
        self.pivot = pivot_params or {'s': 2.3, 'log_nu_b': 9.5, 'f_turb': 0.3}

        # Current parameters (maps on sphere)
        self.s_map = np.full(self.npix, self.pivot['s'])
        self.log_nu_b_map = np.full(self.npix, self.pivot['log_nu_b'])
        self.f_turb_map = np.full(self.npix, self.pivot['f_turb'])

        # Moment basis
        if self.frequencies is not None:
            self.basis = MomentBasis(self.frequencies, self.pivot)
        else:
            self.basis = None

        # LB geometry mask (high latitude)
        self._setup_geometry_mask()

    def _setup_geometry_mask(self):
        """Setup mask for Local Bubble region."""
        l, b = hp.pix2ang(self.nside, np.arange(self.npix), lonlat=True)
        # Local Bubble dominates at |b| > 20 deg
        self.mask = np.abs(b) > 20

    def evaluate(self, pixels=None):
        """
        Evaluate SED for all pixels or subset.

        Returns
        -------
        emission : array, shape (n_pixels, n_frequencies)
            Emission for each pixel
        """
        if pixels is None:
            pixels = np.arange(self.npix)

        n_pixels = len(pixels)
        emission = np.zeros((n_pixels, len(self.frequencies)))

        for i, pix in enumerate(pixels):
            if not self.mask[pix]:
                continue
            emission[i] = self.basis.evaluate(
                s=self.s_map[pix],
                log_nu_b=self.log_nu_b_map[pix]
            )

        return emission

    def fit_pixels(self, data, noise=None, pixels=None):
        """
        Fit parameters to data for each pixel.

        Parameters
        ----------
        data : array, shape (n_pixels, n_frequencies)
            Observed data (residuals after subtracting Galaxy)
        noise : array, shape (n_pixels, n_frequencies)
            Noise standard deviation
        pixels : array or None
            Which pixels to fit (None = all masked pixels)
        """
        if pixels is None:
            pixels = np.where(self.mask)[0]

        fitter = LinearFitter(self.basis)

        for pix in pixels:
            if noise is not None:
                error = noise[pix]
            else:
                error = None

            try:
                result = fitter.fit(data[pix], data_error=error)
                self.s_map[pix] = result['s']
                self.log_nu_b_map[pix] = result['log_nu_b']
                # f_turb not fitted in moment expansion (affects polarization only)
                # Keep current value or use default
            except Exception as e:
                logger.warning(f"Fit failed for pixel {pix}: {e}")


class CylindricalBackground:
    """
    3D cylindrical background volume (Galactic Plane).

    Parameters are defined in Galactic cylindrical coordinates
    (R, phi, z) with R > R_min (200 pc).
    """

    def __init__(self, R_bins=None, z_bins=None, phi_bins=None,
                 frequencies=None, pivot_params=None):
        """
        Parameters
        ----------
        R_bins : array
            Galactocentric radius bins (pc)
        z_bins : array
            Height above plane bins (pc)
        phi_bins : array
            Azimuthal angle bins (rad)
        frequencies : array
            Frequencies in Hz
        pivot_params : dict
            Pivot parameters for moment expansion
        """
        # Default bins
        self.R_bins = R_bins or np.linspace(0.2, 20.0, 50)  # 0.2 to 20 kpc
        self.z_bins = z_bins or np.linspace(-2.0, 2.0, 20)  # -2 to 2 kpc
        self.phi_bins = phi_bins or np.linspace(0, 2*np.pi, 36)  # 10 deg steps

        self.n_R = len(self.R_bins) - 1
        self.n_z = len(self.z_bins) - 1
        self.n_phi = len(self.phi_bins) - 1

        self.frequencies = np.atleast_1d(frequencies) if frequencies is not None else None

        # Default pivot parameters (CI model for Galaxy)
        self.pivot = pivot_params or {'s': 2.5, 'log_nu_b': 9.0, 'f_turb': 0.5}

        # Current parameters (3D grid)
        self.s_cube = np.full((self.n_R, self.n_phi, self.n_z), self.pivot['s'])
        self.log_nu_b_cube = np.full((self.n_R, self.n_phi, self.n_z), self.pivot['log_nu_b'])
        self.f_turb_cube = np.full((self.n_R, self.n_phi, self.n_z), self.pivot['f_turb'])

        # Emissivity (will be computed)
        self.emissivity = None

        # Moment basis
        if self.frequencies is not None:
            self.basis = MomentBasis(self.frequencies, self.pivot)
        else:
            self.basis = None

    def galactic_to_cylindrical(self, l, b, distance):
        """
        Convert Galactic coordinates (l, b, d) to cylindrical (R, phi, z).

        Parameters
        ----------
        l, b : float or array
            Galactic longitude and latitude (deg)
        distance : float or array
            Distance from Sun (pc)

        Returns
        -------
        R, phi, z : arrays
            Cylindrical coordinates (R in pc, phi in rad, z in pc)
        """
        # Sun at (R, phi, z) = (8300 pc, 0, 0)
        R_sun = 8300.0

        l_rad = np.radians(l)
        b_rad = np.radians(b)

        # Cartesian from Sun
        x = distance * np.cos(b_rad) * np.cos(l_rad)
        y = distance * np.cos(b_rad) * np.sin(l_rad)
        z = distance * np.sin(b_rad)

        # Galactic center at origin
        x_gal = R_sun - x
        y_gal = y
        z_gal = z

        # Cylindrical
        R = np.sqrt(x_gal**2 + y_gal**2)
        phi = np.arctan2(y_gal, x_gal)

        return R, phi, z_gal

    def integrate_along_los(self, nside, return_3d=False):
        """
        Integrate emission along line of sight to produce HEALPix map.

        Parameters
        ----------
        nside : int
            Output HEALPix resolution
        return_3d : bool
            If True, also return 3D volume before integration

        Returns
        -------
        emission_map : array, shape (npix, n_frequencies)
            Integrated emission for each pixel
        """
        npix = hp.nside2npix(nside)
        emission_map = np.zeros((npix, len(self.frequencies)))

        # Sample points along each line of sight
        # Distance range: 200 pc to 20 kpc
        d_samples = np.logspace(np.log10(200), np.log10(20000), 100)  # pc

        # Get pixel directions
        theta, phi = hp.pix2ang(nside, np.arange(npix))
        l = np.degrees(phi)
        b = 90 - np.degrees(theta)

        # For each LOS sample
        for i_pix in range(npix):
            if i_pix % 1000 == 0:
                logger.debug(f"Processing pixel {i_pix}/{npix}")

            # Sample along LOS
            los_emission = np.zeros(len(self.frequencies))

            for d in d_samples:
                # Convert to cylindrical
                R, phi_cyl, z = self.galactic_to_cylindrical(l[i_pix], b[i_pix], d)

                # Check if in volume
                if R < self.R_bins[0] or R >= self.R_bins[-1]:
                    continue
                if z < self.z_bins[0] or z >= self.z_bins[-1]:
                    continue

                # Find bin indices
                i_R = np.searchsorted(self.R_bins, R) - 1
                i_phi = np.searchsorted(self.phi_bins, phi_cyl) - 1
                i_z = np.searchsorted(self.z_bins, z) - 1

                # Get parameters at this location
                s = self.s_cube[i_R, i_phi % self.n_phi, i_z]
                log_nu_b = self.log_nu_b_cube[i_R, i_phi % self.n_phi, i_z]
                f_turb = self.f_turb_cube[i_R, i_phi % self.n_phi, i_z]

                # Emissivity from moment expansion
                emiss = self.basis.evaluate(s=s, log_nu_b=log_nu_b, f_turb=f_turb)

                # Integration weight (simple trapezoid)
                dd = np.gradient(d_samples)[d_samples == d][0] if d in d_samples else np.mean(np.gradient(d_samples))
                los_emission += emiss * dd

            emission_map[i_pix] = los_emission

        if return_3d:
            return emission_map, None  # Could return 3D cube
        return emission_map

    def fit_voxels(self, data, noise=None, mask=None):
        """
        Fit parameters to 3D data (from deprojected observations).

        For now, simplified: fit each voxel independently.
        In practice, would use regularization and forward model.
        """
        # Placeholder: would implement proper fitting
        # For now, just smooth current parameters
        from scipy.ndimage import gaussian_filter

        self.s_cube = gaussian_filter(self.s_cube, sigma=1.0)
        self.log_nu_b_cube = gaussian_filter(self.log_nu_b_cube, sigma=1.0)
        self.f_turb_cube = gaussian_filter(self.f_turb_cube, sigma=1.0)


class LightBoxFitter:
    """
    Iterative fitter for Local Bubble (foreground) + Galaxy (background).

    Alternates between:
    1. Fix Galaxy, fit LB to residuals
    2. Fix LB, fit Galaxy to residuals
    """

    def __init__(self, nside=128, frequencies=None, n_iter=10):
        """
        Parameters
        ----------
        nside : int
            HEALPix resolution
        frequencies : array
            Frequencies in Hz
        n_iter : int
            Maximum iterations
        """
        self.nside = nside
        self.npix = hp.nside2npix(nside)
        self.frequencies = np.atleast_1d(frequencies)
        self.n_freq = len(self.frequencies)
        self.n_iter = n_iter

        # Initialize components
        self.lb = SphericalForeground(nside=nside, frequencies=frequencies)
        self.galaxy = CylindricalBackground(frequencies=frequencies)

        # Convergence tracking
        self.chi2_history = []
        self.param_history = []

    def forward_model(self):
        """
        Compute total emission: LB + Galaxy.

        Returns
        -------
        total : array, shape (npix, n_freq)
            Total model emission
        """
        # Get LB foreground
        lb_emission = self.lb.evaluate()

        # Get Galaxy background (integrated)
        gal_emission = self.galaxy.integrate_along_los(self.nside)

        return lb_emission + gal_emission

    def fit_iteration(self, data, noise=None, component='both'):
        """
        Single iteration of fitting.

        Parameters
        ----------
        data : array, shape (npix, n_freq)
            Observed data
        noise : array, shape (npix, n_freq)
            Noise standard deviation
        component : str
            'LB', 'galaxy', or 'both'
        """
        if component in ['LB', 'both']:
            # Fix Galaxy, fit LB
            logger.info("Fitting Local Bubble...")
            galaxy_model = self.galaxy.integrate_along_los(self.nside)
            residuals = data - galaxy_model

            # Fit LB to residuals
            self.lb.fit_pixels(residuals, noise=noise)

        if component in ['galaxy', 'both']:
            # Fix LB, fit Galaxy
            logger.info("Fitting Galaxy...")
            lb_model = self.lb.evaluate()
            residuals = data - lb_model

            # Fit Galaxy to residuals (simplified)
            # In practice: proper tomographic inversion
            self.galaxy.fit_voxels(residuals, noise=noise)

    def compute_chi2(self, data, model, noise=None):
        """Compute chi-squared statistic."""
        residuals = data - model
        if noise is not None:
            chi2 = np.sum((residuals / noise)**2)
        else:
            chi2 = np.sum(residuals**2)
        return chi2

    def fit(self, data, noise=None, tol=1e-3):
        """
        Run iterative fitting until convergence.

        Parameters
        ----------
        data : array, shape (npix, n_freq)
            Observed data
        noise : array, shape (npix, n_freq)
            Noise standard deviation
        tol : float
            Convergence tolerance (relative change in chi2)

        Returns
        -------
        result : dict
            Fitting results
        """
        logger.info(f"Starting Light Box fitting with {self.n_iter} max iterations")

        for i in range(self.n_iter):
            logger.info(f"\n=== Iteration {i+1}/{self.n_iter} ===")

            # Current model
            model = self.forward_model()

            # Compute chi2
            chi2 = self.compute_chi2(data, model, noise)
            self.chi2_history.append(chi2)
            logger.info(f"Chi2 before: {chi2:.2e}")

            # Check convergence
            if i > 0:
                delta = abs(self.chi2_history[-1] - self.chi2_history[-2]) / self.chi2_history[-2]
                logger.info(f"Relative change: {delta:.2e}")
                if delta < tol:
                    logger.info("Converged!")
                    break

            # Alternate fitting
            if i % 2 == 0:
                self.fit_iteration(data, noise, component='LB')
            else:
                self.fit_iteration(data, noise, component='galaxy')

        # Final model
        final_model = self.forward_model()
        final_chi2 = self.compute_chi2(data, final_model, noise)

        return {
            'chi2_history': self.chi2_history,
            'final_chi2': final_chi2,
            'lb_params': {
                's': self.lb.s_map.copy(),
                'log_nu_b': self.lb.log_nu_b_map.copy(),
                'f_turb': self.lb.f_turb_map.copy(),
            },
            'galaxy_params': {
                's': self.galaxy.s_cube.copy(),
                'log_nu_b': self.galaxy.log_nu_b_cube.copy(),
            },
            'final_model': final_model,
        }


def generate_synthetic_data(nside=64, frequencies=None, noise_level=0.1):
    """
    Generate synthetic data for testing.

    Creates:
    - Local Bubble: spherical shell at ~100 pc
    - Galaxy: spiral arm structure
    """
    if frequencies is None:
        frequencies = np.array([23, 44, 70, 100, 143, 217, 353]) * 1e9

    npix = hp.nside2npix(nside)

    # Get coordinates
    l, b = hp.pix2ang(nside, np.arange(npix), lonlat=True)

    # True LB parameters: shell at high latitude
    true_lb_s = np.full(npix, 2.3)
    true_lb_s[np.abs(b) > 30] = 2.8  # North Polar Spur region
    true_lb_log_nu_b = np.full(npix, 9.5)
    true_lb_log_nu_b[np.abs(b) > 30] = 10.2  # Higher break frequency in NPS

    # True Galaxy parameters (simplified)
    # Spiral arms: lower s in arms
    true_gal_s = 2.5 - 0.3 * np.sin(np.radians(l) * 2) * np.exp(-np.abs(np.radians(b)))

    # Generate emission
    lb = SphericalForeground(nside=nside, frequencies=frequencies)
    lb.s_map = true_lb_s
    lb.log_nu_b_map = true_lb_log_nu_b
    lb_emission = lb.evaluate()

    # Simple Galaxy model (just 2D for testing)
    # In real version, would use CylindricalBackground
    gal_emission = np.zeros((npix, len(frequencies)))
    for i in range(npix):
        sed = ci_model(frequencies, true_gal_s[i], 10**9.0)
        gal_emission[i] = sed

    total = lb_emission + gal_emission

    # Add noise
    noise = noise_level * np.std(total) * np.random.randn(*total.shape)
    data = total + noise

    return data, total, lb_emission, gal_emission, frequencies


if __name__ == '__main__':
    # Test the light box fitter
    print("Generating synthetic data...")
    data, truth, lb_true, gal_true, freqs = generate_synthetic_data(nside=32)

    print(f"Data shape: {data.shape}")
    print(f"Frequencies: {freqs/1e9} GHz")

    # Create fitter
    fitter = LightBoxFitter(nside=32, frequencies=freqs, n_iter=5)

    # Fit
    print("\nRunning Light Box fitting...")
    result = fitter.fit(data, noise=None, tol=0.01)

    print("\n" + "="*50)
    print("Fitting complete!")
    print(f"Final chi2: {result['final_chi2']:.2e}")
    print(f"Chi2 history: {result['chi2_history']}")
