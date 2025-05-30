import numpy as np
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)

def visualize_scalar_field(field: np.ndarray, output_path: str) -> None:
    """Visualize a scalar field (2D slice)."""
    try:
        slice_2d = np.abs(field[:, :, field.shape[2]//2, field.shape[3]//2, 0, 0])
        plt.figure(figsize=(8, 6))
        plt.imshow(slice_2d, cmap='viridis')
        plt.colorbar(label='Field Amplitude')
        plt.title("Nugget Scalar Field Slice")
        plt.savefig(output_path)
        plt.close()
        logger.info("Nugget scalar field visualization saved to %s", output_path)
    except Exception as e:
        logger.error("Scalar field visualization failed: %s", e)
        raise

def visualize_rio_field(ricci_scalar: np.ndarray, output_path: str, boundary_factor: float = 1.0,
                        body_positions: list = None) -> None:
    """Visualize the Rio Ricci scalar field (2D slice) with AdS boundary effects."""
    try:
        slice_2d = ricci_scalar[:, :, ricci_scalar.shape[2]//2, ricci_scalar.shape[3]//2, 0, 0]
        plt.figure(figsize=(8, 6))
        plt.imshow(slice_2d, cmap='inferno')
        plt.colorbar(label='Rio Ricci Scalar (Boundary Factor: %.2f)' % boundary_factor)
        plt.title("Rio Ricci Scalar Slice")
        plt.savefig(output_path)
        plt.close()
        dist_sum = (sum(np.sqrt(sum((p1 - p2)**2)) for i, p1 in enumerate(body_positions) 
                        for p2 in body_positions[i+1:]) if body_positions else 0.0)
        logger.info("Rio Ricci scalar visualization saved to %s with boundary_factor=%.2f, body_dist_sum=%.6f", 
                    output_path, boundary_factor, dist_sum)
    except Exception as e:
        logger.error("Rio Ricci scalar visualization failed: %s", e)
        raise

def visualize_boundary_correlations(entanglement_metrics: np.ndarray, boundary_factors: np.ndarray, 
                                    metric_name: str, output_path: str) -> None:
    """Visualize entanglement metrics vs. AdS boundary factors."""
    try:
        plt.figure(figsize=(10, 6))
        plt.scatter(boundary_factors, entanglement_metrics, c='purple', alpha=0.5)
        plt.xlabel("AdS Boundary Factor")
        plt.ylabel(metric_name)
        plt.title(f"{metric_name} vs. AdS Boundary Factor")
        plt.grid(True)
        plt.savefig(output_path)
        plt.close()
        logger.info("Boundary correlations visualization saved to %s", output_path)
    except Exception as e:
        logger.error("Boundary correlations visualization failed: %s", e)
        raise

def visualize_trajectories(trajectories: np.ndarray, output_path: str) -> None:
    """Visualize three-body trajectories in 2D projection."""
    try:
        plt.figure(figsize=(10, 8))
        for i, traj in enumerate(trajectories):
            traj = np.array(traj)
            plt.plot(traj[:, 0], traj[:, 1], label=f'Body {i+1}')
        plt.xlabel("X Position")
        plt.ylabel("Y Position")
        plt.title("Three-Body Trajectories")
        plt.legend()
        plt.grid(True)
        plt.savefig(output_path)
        plt.close()
        logger.info("Trajectories visualization saved to %s", output_path)
    except Exception as e:
        logger.error("Trajectories visualization failed: %s", e)
        raise
