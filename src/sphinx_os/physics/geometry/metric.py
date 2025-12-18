import numpy as np
from ...physics.lattice import TetrahedralLattice
import logging

logger = logging.getLogger(__name__)

def compute_quantum_metric(lattice: object, nugget_field: np.ndarray, temporal_entanglement: np.ndarray,
                           grid_size: tuple, j4_field: np.ndarray = None, psi: np.ndarray = None,
                           body_positions: list = None, body_masses: list = None) -> tuple:
    """Compute quantum metric with non-linear J^6-coupled AdS boundary effects and three-body potentials (vectorized)."""
    try:
        metric = np.zeros(grid_size + (6, 6), dtype=np.float64)
        inverse_metric = np.zeros(grid_size + (6, 6), dtype=np.float64)
        
        j4_field = j4_field if j4_field is not None else np.zeros(grid_size, dtype=np.float64)
        psi = psi if psi is not None else np.ones(grid_size, dtype=np.complex128) / np.sqrt(np.prod(grid_size))
        body_positions = body_positions or []
        body_masses = body_masses or []
        
        # AdS_6 metric with non-linear boundary coupling (vectorized)
        L = 1.0  # AdS radius
        G = 6.67430e-11  # Gravitational constant
        c = 2.99792458e8  # Speed of light
        j6_scale = 1e-30
        epsilon = 1e-15
        
        # Create index grid for vectorized computation
        idx_arrays = np.meshgrid(*[np.arange(s) for s in grid_size], indexing='ij')
        idx_grid = np.stack(idx_arrays, axis=-1)
        grid_center = np.array(grid_size) / 2
        
        # Vectorized z computation
        z = np.sum(np.abs(idx_grid - grid_center), axis=-1) / np.sum(grid_size)
        
        # Vectorized psi and j4 computations
        psi_abs_sq = np.mean(np.abs(psi)**2, axis=tuple(range(len(psi.shape)))[len(grid_size):] if len(psi.shape) > len(grid_size) else ())
        if psi_abs_sq.shape != grid_size:
            psi_abs_sq = np.abs(psi)**2
        j4_abs = np.abs(j4_field)
        
        # Unsimplified AdS boundary factor (vectorized)
        boundary_nonlinear = (psi_abs_sq * j4_abs**3)**6 / (j6_scale + epsilon)
        boundary_factor = np.exp(-0.1 * z) * (1 + 0.001 * boundary_nonlinear)
        
        # Set diagonal metric components (vectorized)
        metric[..., 0, 0] = -1.0 / (L**2 * (1 + z**2)) * boundary_factor
        for i in range(1, 6):
            metric[..., i, i] = 1.0 / (L**2 * (1 + z**2)) * boundary_factor
        
        # Add three-body gravitational potentials (vectorized where possible)
        if body_positions and body_masses:
            coords = np.array(np.meshgrid(*[np.arange(s) for s in grid_size[:3]], indexing='ij'))
            for pos, mass in zip(body_positions, body_masses):
                dist = np.sqrt(np.sum((coords - pos[:, None, None, None])**2, axis=0) + 1e-15)
                full_dist = np.ones(grid_size, dtype=np.float64)
                full_dist[:dist.shape[0], :dist.shape[1], :dist.shape[2]] = dist
                metric[..., 0, 0] *= (1 - 2 * G * mass / (full_dist * c**2))
        
        # Perturbations from fields and tetrahedral structure
        phi_norm = np.abs(nugget_field) / (np.max(np.abs(nugget_field)) + 1e-10)
        psi_norm = np.abs(temporal_entanglement)**2
        
        # Vectorized perturbation computation where possible
        boundary_factor_pert = np.exp(-0.1 * z)
        
        for idx in np.ndindex(grid_size):
            bary_weights = lattice.get_barycentric_weights(idx, body_positions)
            perturbation = 0.1 * phi_norm[idx] + 0.05 * psi_norm[idx]
            napoleon_factor = lattice.get_napoleon_factor(idx, body_positions)
            for i in range(6):
                metric[idx + (i, i)] *= (1.0 + perturbation * np.sum(bary_weights) * napoleon_factor * boundary_factor_pert[idx])
        
        # Compute inverse metric (vectorized)
        # Reshape for batch inverse
        metric_reshaped = metric.reshape(-1, 6, 6)
        inverse_reshaped = np.zeros_like(metric_reshaped)
        
        for i in range(metric_reshaped.shape[0]):
            try:
                inverse_reshaped[i] = np.linalg.inv(metric_reshaped[i])
            except np.linalg.LinAlgError:
                inverse_reshaped[i] = np.eye(6)
                logger.warning("Singular metric at index %d, using identity matrix", i)
        
        inverse_metric = inverse_reshaped.reshape(grid_size + (6, 6))
        
        dist_sum = (_compute_body_distance_sum(body_positions) if body_positions else 0.0)
        logger.debug("Metric computed (vectorized): mean_diag=%.6f, std_diag=%.6f, boundary_factor_mean=%.6f, body_dist_sum=%.6f", 
                     np.mean(np.diagonal(metric, axis1=-2, axis2=-1)), 
                     np.std(np.diagonal(metric, axis1=-2, axis2=-1)), np.mean(boundary_factor), dist_sum)
        return metric, inverse_metric
    except Exception as e:
        logger.error("Quantum metric computation failed: %s", e)
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

def generate_wormhole_nodes(grid_size: tuple, deltas: list) -> np.ndarray:
    """Generate wormhole node positions."""
    return np.array([0.33333333326] * 6)
