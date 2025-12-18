import numpy as np
import logging

logger = logging.getLogger(__name__)

def initialize_graviton_field(grid_size: tuple, deltas: list) -> np.ndarray:
    """Initialize the spin-2 graviton field h_{\mu\nu}."""
    try:
        graviton_field = np.zeros(grid_size + (6, 6), dtype=np.float64)
        for idx in np.ndindex(grid_size):
            graviton_field[idx] = np.random.normal(0, 1e-5, (6, 6))
            graviton_field[idx] = (graviton_field[idx] + graviton_field[idx].T) / 2  # Ensure symmetry
        logger.info("Graviton field initialized with grid size %s", grid_size)
        return graviton_field
    except Exception as e:
        logger.error("Graviton field initialization failed: %s", e)
        raise

def evolve_graviton_field(graviton_field: np.ndarray, grid_size: tuple, deltas: list, dt: float,
                          scalar_field: np.ndarray, ricci_scalar: np.ndarray, 
                          psi: np.ndarray, j4_field: np.ndarray,
                          body_positions: list = None, body_masses: list = None) -> tuple:
    """Evolve the graviton field with three-body sources and unsimplified AdS boundary effects (vectorized)."""
    try:
        dx = deltas[1]
        steps = []
        h = graviton_field
        
        # Compute Laplacian (vectorized)
        laplacian = np.zeros_like(h)
        for dim in range(min(6, len(grid_size))):
            laplacian += np.roll(h, 1, axis=dim) + np.roll(h, -1, axis=dim) - 2 * h
        laplacian /= dx**2
        
        # Initialize source term
        source = np.zeros_like(h)
        j6_scale = 1e-30
        epsilon = 1e-15
        z = np.sum(np.abs(np.array(grid_size) / 2)) / np.sum(grid_size)
        
        # Three-body gravitational source (vectorized where possible)
        if body_positions and body_masses:
            G = 6.67430e-11  # Gravitational constant
            coords = np.array(np.meshgrid(*[np.arange(s) for s in grid_size[:3]], indexing='ij'))
            
            # Compute graviton traces (vectorized)
            graviton_traces = np.trace(h, axis1=-2, axis2=-1)
            boundary_factor = np.exp(-0.1 * z) * (1 + 0.001 * (np.abs(graviton_traces)**6 / (j6_scale + epsilon)))
            
            for pos, mass in zip(body_positions, body_masses):
                dist = np.sqrt(np.sum((coords - pos[:, None, None, None])**2, axis=0) + 1e-15)
                full_dist = np.ones(grid_size, dtype=np.float64)
                full_dist[:dist.shape[0], :dist.shape[1], :dist.shape[2]] = dist
                
                # Broadcast source term
                for i in range(6):
                    source[..., i, i] += G * mass / full_dist * boundary_factor
        
        # Evolve field
        d2h_dt2 = laplacian + source
        new_h = h + dt * (h - np.roll(h, 1, axis=0)) / dt + 0.5 * dt**2 * d2h_dt2
        
        # Ensure symmetry (vectorized)
        new_h = (new_h + np.swapaxes(new_h, -2, -1)) / 2
        
        # Clip for numerical stability
        new_h = np.clip(new_h, -1e3, 1e3)
        
        # Log metrics
        graviton_trace = np.mean(np.trace(new_h, axis1=-2, axis2=-1))
        dist_sum = (_compute_body_distance_sum(body_positions) if body_positions else 0.0)
        steps.append({
            "graviton_trace": graviton_trace,
            "graviton_norm": np.linalg.norm(new_h),
            "body_dist_sum": dist_sum
        })
        
        boundary_factor_final = boundary_factor if body_positions and body_masses else np.exp(-0.1 * z)
        logger.debug("Graviton field evolved (vectorized): trace=%.6f, norm=%.6f, body_dist_sum=%.6f, boundary_factor_mean=%.6f", 
                     steps[-1]["graviton_trace"], steps[-1]["graviton_norm"], 
                     dist_sum, np.mean(boundary_factor_final) if isinstance(boundary_factor_final, np.ndarray) else boundary_factor_final)
        return new_h, steps
    except Exception as e:
        logger.error("Graviton field evolution failed: %s", e)
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
