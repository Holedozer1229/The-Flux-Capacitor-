import numpy as np
import logging
from ...constants import Constants

logger = logging.getLogger(__name__)

def compute_riemann_tensor(metric: np.ndarray, grid_size: tuple, dx: float, 
                           body_positions: list = None) -> np.ndarray:
    """Compute the Riemann curvature tensor R^{\rho}_{\sigma\mu\nu} (vectorized)."""
    try:
        riemann = np.zeros(grid_size + (6, 6, 6, 6), dtype=np.float64)
        christoffel = np.zeros(grid_size + (6, 6, 6), dtype=np.float64)
        
        # Compute Christoffel symbols (first kind) - vectorized over spatial dimensions
        for rho, mu, nu in np.ndindex((6, 6, 6)):
            for sigma in range(6):
                # Vectorized computation across all spatial points
                christoffel[..., rho, mu, nu] += 0.5 * (
                    np.roll(metric[..., mu, sigma], -1, axis=mu % len(grid_size)) / dx -
                    np.roll(metric[..., nu, sigma], -1, axis=nu % len(grid_size)) / dx +
                    np.roll(metric[..., mu, nu], -1, axis=sigma % len(grid_size)) / dx
                )
        
        # Compute Riemann tensor - vectorized over spatial dimensions
        for rho, sigma, mu, nu in np.ndindex((6, 6, 6, 6)):
            riemann[..., rho, sigma, mu, nu] = (
                np.roll(christoffel[..., rho, nu, sigma], -1, axis=mu % len(grid_size)) / dx -
                np.roll(christoffel[..., rho, mu, sigma], -1, axis=nu % len(grid_size)) / dx
            )
            # Vectorized contraction over lambda
            for lambda_idx in range(6):
                riemann[..., rho, sigma, mu, nu] += (
                    christoffel[..., rho, mu, lambda_idx] * 
                    christoffel[..., lambda_idx, nu, sigma] -
                    christoffel[..., rho, nu, lambda_idx] * 
                    christoffel[..., lambda_idx, mu, sigma]
                )
        
        logger.debug("Riemann tensor computed (vectorized): mean=%.6e, std=%.6e", 
                     np.mean(riemann), np.std(riemann))
        return riemann
    except Exception as e:
        logger.error("Riemann tensor computation failed: %s", e)
        raise

def compute_curvature(metric: np.ndarray, inverse_metric: np.ndarray, grid_size: tuple, 
                      dx: float, scalar_field: np.ndarray = None, 
                      body_positions: list = None) -> tuple:
    """Compute Ricci tensor and Rio Ricci scalar with unsimplified AdS boundary effects (vectorized)."""
    try:
        riemann = compute_riemann_tensor(metric, grid_size, dx, body_positions)
        ricci_tensor = np.zeros(grid_size + (6, 6), dtype=np.float64)
        
        # Compute Ricci tensor - vectorized contraction
        for mu, nu in np.ndindex((6, 6)):
            for lambda_idx in range(6):
                ricci_tensor[..., mu, nu] += riemann[..., lambda_idx, mu, lambda_idx, nu]
        
        # Compute Rio Ricci scalar - vectorized Einstein summation
        ricci_scalar = np.einsum('...ij,...ij->...', inverse_metric, ricci_tensor)
        
        # Apply AdS boundary factor (vectorized)
        j6_scale = 1e-30
        epsilon = 1e-15
        z = np.sum(np.abs(np.array(grid_size) / 2)) / np.sum(grid_size)
        
        if scalar_field is not None:
            phi_norm = np.abs(scalar_field) / (np.max(np.abs(scalar_field)) + 1e-10)
        else:
            phi_norm = np.ones(grid_size, dtype=np.float64)
        
        boundary_factor = np.exp(-0.1 * z) * (1 + 0.001 * (phi_norm**6 / (j6_scale + epsilon)))
        ricci_scalar *= boundary_factor
        
        # Three-body curvature perturbation (vectorized where possible)
        if body_positions:
            G = 6.67430e-11
            # Create coordinate grid
            coords = np.array(np.meshgrid(*[np.arange(s) for s in grid_size[:3]], indexing='ij'))
            
            for pos in body_positions:
                # Vectorized distance computation
                dist = np.sqrt(np.sum((coords - pos[:, None, None, None])**2, axis=0) + 1e-15)
                # Broadcast to full grid size
                full_dist = np.ones(grid_size, dtype=np.float64)
                full_dist[:dist.shape[0], :dist.shape[1], :dist.shape[2]] = dist
                ricci_scalar += G / full_dist * 0.01
        
        # Clip for stability
        ricci_scalar = np.clip(ricci_scalar, -1e5, 1e5)
        
        # Log metrics
        ricci_mean = np.mean(np.abs(ricci_scalar))
        ricci_std = np.std(ricci_scalar)
        dist_sum = (_compute_body_distance_sum(body_positions) if body_positions else 0.0)
        logger.debug("Curvature computed (vectorized): ricci_mean=%.6f, ricci_std=%.6f, boundary_factor_mean=%.6f, body_dist_sum=%.6f", 
                     ricci_mean, ricci_std, np.mean(boundary_factor), dist_sum)
        return ricci_tensor, ricci_scalar
    except Exception as e:
        logger.error("Curvature computation failed: %s", e)
        raise


def _compute_body_distance_sum(body_positions: list) -> float:
    """Compute sum of pairwise distances between bodies (optimized)."""
    if not body_positions:
        return 0.0
    positions = np.array(body_positions)
    n = len(positions)
    dist_sum = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            dist_sum += np.linalg.norm(positions[i] - positions[j])
    return dist_sum
