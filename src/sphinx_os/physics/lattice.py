import numpy as np
import logging

logger = logging.getLogger(__name__)

class TetrahedralLattice:
    """6D tetrahedral lattice for spacetime modeling."""
    
    def __init__(self, grid_size: tuple):
        """Initialize the 6D tetrahedral lattice."""
        self.grid_size = grid_size
        self.dx = 1.0 / np.max(grid_size)  # Lattice spacing
        self.neighbors = [(1,0,0,0,0,0), (-1,0,0,0,0,0), (0,1,0,0,0,0), (0,-1,0,0,0,0),
                         (0,0,1,0,0,0), (0,0,-1,0,0,0), (0,0,0,1,0,0), (0,0,0,-1,0,0),
                         (0,0,0,0,1,0), (0,0,0,0,-1,0), (0,0,0,0,0,1), (0,0,0,0,0,-1)]
        logger.info("Tetrahedral lattice initialized with grid size %s", grid_size)
    
    def get_neighbors(self, idx: tuple) -> list:
        """Get neighboring lattice points."""
        try:
            neighbors = []
            for offset in self.neighbors:
                neighbor = tuple(i + o for i, o in zip(idx, offset))
                if all(0 <= n < s for n, s in zip(neighbor, self.grid_size)):
                    neighbors.append(neighbor)
            logger.debug("Neighbors retrieved for idx %s: %d neighbors", idx, len(neighbors))
            return neighbors
        except Exception as e:
            logger.error("Neighbor retrieval failed: %s", e)
            raise
    
    def get_barycentric_weights(self, idx: tuple, body_positions: list = None) -> np.ndarray:
        """Compute barycentric interpolation weights with unsimplified AdS boundary effects."""
        try:
            weights = np.array([0.25, 0.25, 0.25, 0.25])  # Default tetrahedral weights
            z = np.sum(np.abs(np.array(self.grid_size) / 2)) / np.sum(self.grid_size)
            j6_scale = 1e-30
            epsilon = 1e-15
            idx_sum = np.sum(idx)
            
            # Unsimplified AdS boundary factor
            boundary_factor = np.exp(-0.1 * z) * (1 + 0.001 * (idx_sum**6 / (j6_scale + epsilon)))
            weights *= boundary_factor
            
            # Three-body modulation
            if body_positions:
                G = 6.67430e-11
                for pos in body_positions:
                    dist = np.sqrt(sum((np.array(idx[:3]) - pos)**2) + 1e-15)
                    weights *= (1 + 0.01 * G / dist)  # Small perturbation
            
            weights /= np.sum(weights)  # Normalize
            logger.debug("Barycentric weights computed for idx %s: %s, boundary_factor=%.6f", 
                         idx, weights, boundary_factor)
            return weights
        except Exception as e:
            logger.error("Barycentric weights computation failed: %s", e)
            raise
    
    def get_napoleon_factor(self, idx: tuple, body_positions: list = None) -> float:
        """Compute Napoleon’s theorem factor with three-body modulation."""
        try:
            idx_sum = np.sum(idx)
            napoleon_factor = 1 + 0.05 * np.cos(3 * idx_sum)
            
            # Three-body modulation
            from ..utils.math_utils import compute_body_distance_sum
            
            dist_sum = 0.0
            if body_positions:
                dist_sum = compute_body_distance_sum(body_positions)
                napoleon_factor *= np.exp(-0.01 * dist_sum)  # Modulate amplitude
            
            logger.debug("Napoleon factor computed for idx %s: %.6f, body_dist_sum=%.6f", 
                         idx, napoleon_factor, dist_sum)
            return napoleon_factor
        except Exception as e:
            logger.error("Napoleon factor computation failed: %s", e)
            raise
