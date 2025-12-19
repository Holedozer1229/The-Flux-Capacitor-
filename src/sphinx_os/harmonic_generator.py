import numpy as np
from scipy.signal import find_peaks, correlate
from ..utils.math_utils import compute_j6_potential
from ..constants import CONFIG
import logging

logger = logging.getLogger(__name__)

class HarmonicGenerator:
    """Generate and analyze harmonics with non-linear J^6 coupling."""
    
    def __init__(self, sample_rate: int, kappa_j6: float):
        self.sample_rate = sample_rate
        self.kappa_j6 = kappa_j6
        self.kappa_j6_eff = CONFIG["kappa_j6_eff"]
        self.j6_scaling_factor = CONFIG["j6_scaling_factor"]
        self.epsilon = CONFIG["epsilon"]
        self.omega_res = CONFIG["resonance_frequency"] * 2 * np.pi
        logger.info("Harmonic generator initialized with sample_rate=%d", sample_rate)
    
    def generate_harmonics(self, phi: float, j4: float, psi: np.ndarray, ricci_scalar: np.ndarray,
                           graviton_field: np.ndarray = None, boundary_factor: float = 1.0,
                           body_positions: list = None) -> np.ndarray:
        """Generate harmonics with non-linear J^6-coupled graviton and boundary effects."""
        try:
            from .utils.math_utils import compute_body_distance_sum
            
            V_j6, _ = compute_j6_potential(
                np.array([phi]), np.array([j4]), psi, ricci_scalar, graviton_field,
                self.kappa_j6, self.kappa_j6_eff, self.j6_scaling_factor, self.epsilon,
                self.omega_res, boundary_factor, body_positions=body_positions
            )
            harmonics = np.sin(2 * np.pi * V_j6 / self.sample_rate)
            harmonics = np.clip(harmonics, -1.0, 1.0)
            dist_sum = compute_body_distance_sum(body_positions) if body_positions else 0.0
            logger.debug("Harmonics generated: mean=%.6f, boundary_factor=%.6f, body_dist_sum=%.6f", 
                         np.mean(harmonics), boundary_factor, dist_sum)
            return harmonics
        except Exception as e:
            logger.error("Harmonics generation failed: %s", e)
            raise
    
    def analyze_harmonics(self, harmonics: np.ndarray, output_path: str, body_positions: list = None) -> list:
        """Analyze harmonic frequencies with three-body influence."""
        try:
            from .utils.math_utils import compute_body_distance_sum
            
            freqs = np.fft.fftfreq(len(harmonics), 1 / self.sample_rate)
            spectrum = np.abs(np.fft.fft(harmonics))
            peaks, _ = find_peaks(spectrum[:len(spectrum)//2], height=np.max(spectrum)/10)
            peak_freqs = freqs[peaks]
            peak_freqs = peak_freqs[peak_freqs > 0]
            dist_sum = compute_body_distance_sum(body_positions) if body_positions else 0.0
            logger.info("Harmonic peaks detected: %s, body_dist_sum=%.6f", peak_freqs, dist_sum)
            return peak_freqs.tolist()
        except Exception as e:
            logger.error("Harmonic analysis failed: %s", e)
            raise
    
    def analyze_delays(self, output: np.ndarray, input_signal: np.ndarray) -> float:
        """Analyze delays via cross-correlation."""
        try:
            correlation = correlate(output, input_signal, mode='full')
            lags = np.arange(-len(input_signal) + 1, len(output))
            delay_samples = lags[np.argmax(correlation)]
            delay = delay_samples / self.sample_rate
            logger.debug("Delay analyzed: %.6f s", delay)
            return delay
        except Exception as e:
            logger.error("Delay analysis failed: %s", e)
            raise
    
    def analyze_j6_potential(self, phi: float, j4: float, psi: np.ndarray, ricci_scalar: np.ndarray,
                             graviton_field: np.ndarray, boundary_factor: float) -> dict:
        """Analyze J^6 potential metrics."""
        try:
            V_j6, dV_j6_dphi = compute_j6_potential(
                np.array([phi]), np.array([j4]), psi, ricci_scalar, graviton_field,
                self.kappa_j6, self.kappa_j6_eff, self.j6_scaling_factor, self.epsilon,
                self.omega_res, boundary_factor
            )
            stats = {
                'V_j6_mean': np.mean(V_j6),
                'dV_j6_dphi_mean': np.mean(np.abs(dV_j6_dphi)),
                'rio_mean': np.mean(ricci_scalar),
                'graviton_trace': np.mean(np.trace(graviton_field, axis1=-2, axis2=-1)),
                'graviton_nonlinear': np.abs(np.mean(np.trace(graviton_field, axis1=-2, axis2=-1)))**6 / (1e-30 + 1e-15)
            }
            logger.debug("J^6 potential stats: %s", stats)
            return stats
        except Exception as e:
            logger.error("J^6 potential analysis failed: %s", e)
            raise
