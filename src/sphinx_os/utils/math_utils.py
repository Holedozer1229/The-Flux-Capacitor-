import numpy as np
import logging
from functools import lru_cache

logger = logging.getLogger(__name__)


@lru_cache(maxsize=32)
def _compute_body_distance_sum_cached(body_positions_tuple: tuple) -> float:
    """Compute sum of pairwise distances between bodies (cached for repeated calls)."""
    if not body_positions_tuple:
        return 0.0
    positions = np.array(body_positions_tuple)
    n = len(positions)
    dist_sum = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            dist_sum += np.linalg.norm(positions[i] - positions[j])
    return dist_sum


def compute_body_distance_sum(body_positions: list) -> float:
    """Compute sum of pairwise distances between bodies (with caching)."""
    if not body_positions:
        return 0.0
    # Convert to tuple of tuples for caching
    body_positions_tuple = tuple(tuple(pos) for pos in body_positions)
    return _compute_body_distance_sum_cached(body_positions_tuple)


def compute_j6_potential(phi: np.ndarray, j4: np.ndarray, psi: np.ndarray, ricci_scalar: np.ndarray,
                         graviton_field: np.ndarray = None, kappa_j6: float = 1.0, 
                         kappa_j6_eff: float = 1e-33, j6_scaling_factor: float = 1e-30,
                         epsilon: float = 1e-15, omega_res: float = 2 * np.pi * 1e6,
                         boundary_factor: float = 1.0, body_positions: list = None,
                         body_masses: list = None) -> tuple:
    """Compute the unified J^6 coupling potential with non-linear graviton, AdS boundary, and three-body effects."""
    try:
        phi_abs = np.abs(phi)
        denom = 1 + 0.01 * phi_abs
        phi_term = phi_abs**6 * np.sin(phi) / denom
        
        j4_abs = np.abs(j4)
        j4_term = (j4_abs**3) / (j6_scaling_factor + epsilon)
        psi_abs_sq = np.abs(psi)**2
        
        # Validate and clip Rio Ricci scalar
        ricci_scalar = np.clip(ricci_scalar, -1e5, 1e5)
        ricci_mean = np.mean(np.abs(ricci_scalar)) if ricci_scalar.size > 0 else 1.0
        ricci_std = np.std(ricci_scalar) if ricci_scalar.size > 0 else 0.0
        
        # Non-linear graviton coupling
        graviton_trace = np.mean(np.trace(graviton_field, axis1=-2, axis2=-1)) if graviton_field is not None else 0.0
        graviton_nonlinear = np.abs(graviton_trace)**6 / (j6_scaling_factor + epsilon)  # J^6-like term
        graviton_factor = 1 + 0.01 * graviton_trace + 0.001 * graviton_nonlinear  # Combined linear and non-linear
        
        # Three-body influence (vectorized where possible)
        if body_positions and body_masses:
            G = 6.67430e-11  # Gravitational constant
            # For small phi arrays, vectorization is more efficient
            if phi.size < 1000:
                coords = np.array(np.meshgrid(*[np.arange(s) for s in phi.shape[:3]], indexing='ij'))
                for pos, mass in zip(body_positions, body_masses):
                    # Safely reshape pos for broadcasting
                    pos_reshaped = np.array(pos).reshape(-1, 1, 1, 1)
                    dist = np.sqrt(np.sum((coords - pos_reshaped)**2, axis=0) + 1e-15)
                    # Expand to full phi shape if needed
                    if phi.ndim > 3:
                        dist_expanded = np.expand_dims(dist, axis=tuple(range(3, phi.ndim)))
                        dist_expanded = np.broadcast_to(dist_expanded, phi.shape)
                    else:
                        dist_expanded = dist
                    phi_term += G * mass / dist_expanded
            else:
                # Fallback for very large arrays
                for idx in np.ndindex(phi.shape):
                    for pos, mass in zip(body_positions, body_masses):
                        dist = np.sqrt(sum((np.array(idx[:3]) - pos)**2) + 1e-15)
                        phi_term[idx] += G * mass / dist
        
        # Barycentric interpolation for Rio smoothness
        bary_weights = np.array([0.25, 0.25, 0.25, 0.25])
        interpolated_ricci = np.sum(bary_weights * ricci_mean)
        
        # Napoleon’s theorem-inspired modulation
        idx_sum = np.sum(np.indices(ricci_scalar.shape), axis=0) if ricci_scalar.size > 0 else 0
        napoleon_factor = 1 + 0.05 * np.cos(3 * idx_sum)
        
        # Non-linear AdS boundary effect
        boundary_nonlinear = boundary_factor**6 / (j6_scaling_factor + epsilon)  # J^6-like boundary term
        effective_boundary = boundary_factor * (1 + 0.001 * boundary_nonlinear)
        interpolated_ricci *= napoleon_factor * effective_boundary
        
        if not np.isfinite(interpolated_ricci):
            interpolated_ricci = 1.0
            logger.warning("Non-finite Rio Ricci scalar mean, defaulting to 1.0")
        
        modulation = psi_abs_sq * interpolated_ricci * graviton_factor / (2.72 * omega_res)
        em_term = kappa_j6_eff * j4_term * modulation
        
        V_j6 = kappa_j6 * phi_term + em_term
        
        dV_j6_dphi = kappa_j6 * (
            6 * phi_abs**5 * np.sin(phi) / denom +
            phi_abs**6 * (np.cos(phi) * denom - 0.01 * np.sin(phi) * np.sign(phi)) / denom**2
        )
        
        dist_sum = compute_body_distance_sum(body_positions) if body_positions else 0.0
        logger.debug("J^6 potential: mean=%.6e, rio_mean=%.6f, rio_std=%.6f, graviton_trace=%.6f, graviton_nonlinear=%.6e, boundary_factor=%.6f, boundary_nonlinear=%.6e, body_dist_sum=%.6f", 
                     np.mean(V_j6), ricci_mean, ricci_std, graviton_trace, graviton_nonlinear, boundary_factor, boundary_nonlinear, dist_sum)
        return V_j6, dV_j6_dphi
    except Exception as e:
        logger.error("J^6 potential computation failed: %s", e)
        raise
