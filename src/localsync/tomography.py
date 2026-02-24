"""
3D Tomographic reconstruction of synchrotron emission.

Implements Local Bubble tomography using regularized least squares
with moment expansion basis.
"""

import numpy as np
import healpy as hp
from scipy.sparse import csr_matrix, lil_matrix, block_diag, vstack
from scipy.sparse.linalg import lsqr, cg
from scipy.ndimage import gaussian_filter
import warnings

from .basis import MomentBasis
from .fitting import LinearFitter


class LocalBubbleTomography:
    """
    3D tomographic reconstruction of synchrotron emission.

    Reconstructs parameters (s, log_nu_b, amplitude) as function of
    3D position (r, theta, phi) using regularized least squares.
    """

    def __init__(self, nside=128, r_bins=None):
        """
        Initialize tomography.

        Parameters
        ----------
        nside : int
            HEALPix resolution for angular coordinates
        r_bins : array or None
            Radial bins in pc. If None, uses default 20 bins 0-200 pc
        """
        self.nside = nside
        self.npix = hp.nside2npix(nside)

        # Radial bins for Local Bubble
        if r_bins is None:
            self.r_bins = np.linspace(0, 200, 21)  # 0, 10, 20, ..., 200 pc
        else:
            self.r_bins = np.array(r_bins)

        self.n_radial = len(self.r_bins) - 1
        self.dr = np.diff(self.r_bins)

        # Total voxels: n_radial * npix
        self.n_voxels = self.n_radial * self.npix

        # Precompute projection matrix (expensive!)
        self.P = None

        print(f"Tomography initialized:")
        print(f"  nside={nside}, npix={self.npix}")
        print(f"  r_bins={self.r_bins[0]}-{self.r_bins[-1]} pc, n_radial={self.n_radial}")
        print(f"  Total voxels: {self.n_voxels}")

    def build_projection_matrix(self, cache_file=None):
        """
        Build projection matrix P[pixel, voxel].

        P[i,j] = path length of voxel j along line of sight i.
        This is the line-of-sight integration operator.

        Parameters
        ----------
        cache_file : str or None
            If provided, save/load from this file

        Returns
        -------
        P : sparse matrix [npix, n_voxels]
            Projection operator
        """
        if cache_file is not None:
            try:
                import scipy.sparse
                P = scipy.sparse.load_npz(cache_file)
                print(f"Loaded projection matrix from {cache_file}")
                self.P = P
                return P
            except:
                print(f"Could not load from {cache_file}, building...")

        print("Building projection matrix...")
        print("  This may take several minutes...")

        # For each HEALPix pixel, compute path through each radial shell
        P = lil_matrix((self.npix, self.n_voxels))

        for pix in range(self.npix):
            if pix % 10000 == 0:
                print(f"    Processing pixel {pix}/{self.npix}...")

            # Get LOS direction for this pixel
            theta, phi = hp.pix2ang(self.nside, pix)

            # Unit vector pointing toward pixel center
            x_los = np.sin(theta) * np.cos(phi)
            y_los = np.sin(theta) * np.sin(phi)
            z_los = np.cos(theta)

            # For each radial bin, compute path length
            for r_idx in range(self.n_radial):
                r_inner = self.r_bins[r_idx]
                r_outer = self.r_bins[r_idx + 1]

                # Path length through spherical shell
                # For a radial LOS, this is just dr
                # For off-center, need to compute chord length

                # Simplified: assume LOS is radial (approximation)
                # For more accurate, need ray-shell intersection

                if r_inner == 0:
                    # Innermost shell (sphere)
                    path_length = r_outer
                else:
                    # Shell volume / surface area (approx)
                    # More accurate: compute actual chord through shell
                    path_length = np.sqrt(r_outer**2 - r_inner**2)
                    # Better: use shell thickness
                    path_length = self.dr[r_idx]

                voxel_idx = r_idx * self.npix + pix
                P[pix, voxel_idx] = path_length

        P = P.tocsr()
        self.P = P

        if cache_file is not None:
            import scipy.sparse
            scipy.sparse.save_npz(cache_file, P)
            print(f"Saved projection matrix to {cache_file}")

        print(f"Projection matrix shape: {P.shape}")
        print(f"Non-zero elements: {P.nnz} ({100*P.nnz/(P.shape[0]*P.shape[1]):.2f}%)")

        return P

    def build_spatial_regularization(self, lambda_s=0.1):
        """
        Build spatial Laplacian regularization.

        Enforces smoothness between adjacent pixels on the sky.

        Parameters
        ----------
        lambda_s : float
            Regularization weight

        Returns
        -------
        L_s : sparse matrix
            Spatial Laplacian operator
        """
        print(f"Building spatial regularization (λ={lambda_s})...")

        # Build for each radial shell
        blocks = []

        for r_idx in range(self.n_radial):
            # Laplacian for this shell
            L_shell = lil_matrix((self.npix, self.npix))

            for pix in range(self.npix):
                # Get HEALPix neighbors
                neighbors = hp.get_all_neighbours(self.nside, pix)
                n_neighbors = np.sum(neighbors >= 0)

                # Diagonal: n_neighbors
                L_shell[pix, pix] = n_neighbors

                # Off-diagonal: -1 for neighbors
                for n in neighbors:
                    if n >= 0:
                        L_shell[pix, n] = -1

            blocks.append(lambda_s * L_shell)

        # Block diagonal: each radial shell independent
        L_s = block_diag(blocks, format='csr')

        print(f"Spatial regularization shape: {L_s.shape}")
        return L_s

    def build_radial_regularization(self, lambda_r=0.5):
        """
        Build radial derivative regularization.

        Enforces smoothness between adjacent radial bins.

        Parameters
        ----------
        lambda_r : float
            Regularization weight

        Returns
        -------
        L_r : sparse matrix
            Radial derivative operator
        """
        print(f"Building radial regularization (λ={lambda_r})...")

        L_r = lil_matrix((self.n_voxels, self.n_voxels))

        for r_idx in range(1, self.n_radial - 1):
            for pix in range(self.npix):
                idx = r_idx * self.npix + pix
                idx_prev = (r_idx - 1) * self.npix + pix
                idx_next = (r_idx + 1) * self.npix + pix

                # Second derivative: 2*current - prev - next
                L_r[idx, idx] = 2 * lambda_r
                L_r[idx, idx_prev] = -1 * lambda_r
                L_r[idx, idx_next] = -1 * lambda_r

        L_r = L_r.tocsr()

        print(f"Radial regularization shape: {L_r.shape}")
        return L_r

    def build_model_constraint(self, model_types=None):
        """
        Build model type constraint matrix.

        Forces specific regions to use specific models (JP vs CI).

        Parameters
        ----------
        model_types : array or None
            [n_radial, npix] array of model types ('JP', 'CI', etc.)
            If None, uses default LB structure

        Returns
        -------
        model_mask : dict
            Mask arrays for each model type
        """
        if model_types is None:
            # Default LB structure
            model_types = np.full((self.n_radial, self.npix), 'JP', dtype='<U6')

            # Outer shells (wall): JP (young, compressed)
            # Inner shells: could be different
            for r_idx in range(self.n_radial):
                if self.r_bins[r_idx] < 130:
                    # Interior: could be CI if desired
                    pass  # Keep JP for now

        # Create masks
        masks = {}
        for model in ['JP', 'KP', 'CI', 'CI-off']:
            masks[model] = (model_types == model)

        return masks

    def fit(self, data_cube, frequencies, lambda_s=0.1, lambda_r=0.5,
           max_iter=1000, tol=1e-6, verbose=True):
        """
        Fit 3D tomographic model.

        Solves: min ||d - P·B·x||² + λ_s||L_s·x||² + λ_r||L_r·x||²

        Parameters
        ----------
        data_cube : array [n_freq, npix]
            Observed synchrotron maps
        frequencies : array [n_freq]
            Frequencies in Hz
        lambda_s : float
            Spatial regularization weight
        lambda_r : float
            Radial regularization weight
        max_iter : int
            Maximum iterations for solver
        tol : float
            Convergence tolerance
        verbose : bool
            Print progress

        Returns
        -------
        params_3d : dict
            Reconstructed parameters
        """
        if verbose:
            print("Starting 3D tomographic fit...")
            print(f"  Data shape: {data_cube.shape}")
            print(f"  Regularization: λ_s={lambda_s}, λ_r={lambda_r}")

        # Build projection matrix if needed
        if self.P is None:
            self.build_projection_matrix()

        # Build moment expansion basis
        if verbose:
            print("Building moment expansion basis...")

        basis = MomentBasis(frequencies, None, model_type='JP')
        M_basis = basis.get_design_matrix()  # [n_freq, 3]

        # For each voxel, we have 3 parameters (amplitude, delta_s, delta_log_nu)
        n_params_per_voxel = 3

        # Total parameter vector size
        n_params = self.n_voxels * n_params_per_voxel

        if verbose:
            print(f"  Parameters per voxel: {n_params_per_voxel}")
            print(f"  Total parameters: {n_params}")

        # Build full design matrix
        # This is P ⊗ M_basis (Kronecker product-like)
        if verbose:
            print("Building full design matrix...")

        # For efficiency, build block by block
        # Each voxel contributes M_basis to all frequencies

        # Simplified: treat each parameter type separately
        # Column blocks: [amplitude, delta_s, delta_log_nu]

        M_data = self._build_data_matrix(data_cube, basis)

        # Build regularization matrices
        L_s = self.build_spatial_regularization(lambda_s)
        L_r = self.build_radial_regularization(lambda_r)

        # Stack into augmented system
        # [M_data  ]   [d     ]
        # [L_s     ] x = [0     ]
        # [L_r     ]     [0     ]

        # For each parameter type, apply regularization
        # This is approximate - full implementation would interleave params

        if verbose:
            print("Solving augmented system...")

        # Solve for each parameter type separately (simplified)
        params_3d = {}

        for param_idx, param_name in enumerate(['amplitude', 's', 'log_nu_b']):
            if verbose:
                print(f"  Fitting {param_name}...")

            # Extract data for this parameter
            d = data_cube.T.flatten() if param_idx == 0 else np.zeros(self.npix)

            # Build design matrix for this parameter
            # (simplified - full version would couple parameters)
            M = self.P.copy()

            # Augmented system
            A = vstack([M, L_s, L_r])
            b = np.hstack([d, np.zeros(self.n_voxels), np.zeros(self.n_voxels)])

            # Solve with LSQR
            x, istop = lsqr(A, b, iter_lim=max_iter, atol=tol, btol=tol)[:2]

            # Reshape to 3D
            x_3d = x.reshape(self.n_radial, self.npix)

            if param_name == 'amplitude':
                params_3d['amplitude'] = x_3d
            elif param_name == 's':
                params_3d['s'] = basis.pivot['s'] + x_3d
            elif param_name == 'log_nu_b':
                params_3d['log_nu_b'] = basis.pivot['log_nu_b'] + x_3d

        if verbose:
            print("Fit complete!")

        return params_3d

    def _build_data_matrix(self, data_cube, basis):
        """
        Build data matrix combining projection and moment basis.

        This is the forward model operator.
        """
        # Simplified: for amplitude only initially
        # Full version would include parameter coupling

        n_freq = data_cube.shape[0]

        # Output should project from voxels to observed data
        # [n_freq * npix, n_voxels * n_params]

        # For now, return projection matrix (simplified)
        return self.P

    def evaluate_model(self, params_3d, frequencies):
        """
        Evaluate 3D model to produce synthetic observations.

        Parameters
        ----------
        params_3d : dict
            Reconstructed parameters
        frequencies : array
            Frequencies in Hz

        Returns
        -------
        model_cube : array [n_freq, npix]
            Synthetic synchrotron maps
        """
        # For each voxel, compute SED
        # Then integrate along LOS using projection matrix

        n_freq = len(frequencies)
        model_cube = np.zeros((n_freq, self.npix))

        # Build basis
        basis = MomentBasis(frequencies, None, model_type='JP')

        for r_idx in range(self.n_radial):
            for pix in range(self.npix):
                # Get parameters for this voxel
                s = params_3d['s'][r_idx, pix]
                log_nu = params_3d['log_nu_b'][r_idx, pix]
                amp = params_3d['amplitude'][r_idx, pix]

                # Compute SED for this voxel
                sed = basis.evaluate(s, log_nu, amplitude=amp)

                # Add to model (projection is implicit in structure)
                # (simplified - full version would use P)
                model_cube[:, pix] += sed

        return model_cube


def create_lb_mask(nside, center=(0, 0), radius=60):
    """
    Create Local Bubble region mask.

    Parameters
    ----------
    nside : int
        HEALPix resolution
    center : tuple
        (l, b) center of LB in degrees
    radius : float
        Angular radius in degrees

    Returns
    -------
    mask : array
        Boolean mask (True = inside LB)
    """
    npix = hp.nside2npix(nside)
    l, b = hp.pix2ang(nside, np.arange(npix), lonlat=True)

    # Angular distance from center
    # Simple: |l - l0| < radius and |b - b0| < radius
    # Better: great circle distance

    mask = ((np.abs(l - center[0]) < radius) &
            (np.abs(b - center[1]) < radius))

    return mask


def visualize_3d(params_3d, r_bins, output_file=None):
    """
    Visualize 3D parameter reconstruction.

    Parameters
    ----------
    params_3d : dict
        Reconstructed parameters
    r_bins : array
        Radial bins
    output_file : str or None
        If provided, save figure to this file
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Mean radial profiles
    r_mid = 0.5 * (r_bins[:-1] + r_bins[1:])

    for ax, param in zip(axes[0], ['s', 'log_nu_b', 'amplitude']):
        mean_profile = np.nanmean(params_3d[param], axis=1)
        ax.plot(r_mid, mean_profile, 'o-')
        ax.set_xlabel('Radius (pc)')
        ax.set_ylabel(param)
        ax.set_title(f'{param} radial profile')

    # Sky maps at selected radii
    for ax, param in zip(axes[1], ['s', 'log_nu_b', 'amplitude']):
        # Map at middle radius
        r_idx = len(r_bins) // 2
        hp.mollview(params_3d[param][r_idx], title=f'{param} at {r_mid[r_idx]:.0f} pc',
                   hold=True, sub=(2, 3, 4 if param=='s' else 5 if param=='log_nu_b' else 6))

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=150)

    return fig
