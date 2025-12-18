#!/usr/bin/env python3
import logging
import numpy as np
from datetime import datetime
import os
import sounddevice as sd
from scipy import stats
from sphinx_os.anubis_core import UnifiedSpacetimeSimulator
from sphinx_os.qubit_fabric import QuantumCircuit
from sphinx_os.harmonic_generator import HarmonicGenerator
from sphinx_os.arduino_interface import ArduinoInterface
from sphinx_os.unified_toe import UnifiedTOE
from sphinx_os.entanglement_cache import EntanglementCache
from sphinx_os.constants import Constants
from sphinx_os.plotting import plot_fft, plot_entanglement_entropy, plot_mobius_spiral, plot_tetrahedron, plot_j6_validation, plot_rio_validation, plot_graviton_field
from sphinx_os.visualization.visualize import visualize_rio_field, visualize_boundary_correlations
from sphinx_os.utils.math_utils import compute_body_distance_sum
from interface.gui import FluxGUI
from itertools import product

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('flux_capacitor.log'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class FluxCapacitor:
    def __init__(
        self,
        grid_size: tuple = Constants.GRID_SIZE,
        lambda_eigen: float = Constants.LAMBDA_EIGEN,
        port: str = '/dev/ttyUSB0',
        sample_rate: int = Constants.SAMPLE_RATE,
        duration: float = 10.0,
        kappa_j6: float = Constants.KAPPA_J6,
        num_qubits: int = Constants.NUM_QUBITS,
        wormhole_nodes: list = Constants.WORMHOLE_NODES
    ):
        self.grid_size = grid_size
        self.lambda_eigen = lambda_eigen
        self.sample_rate = sample_rate
        self.duration = duration
        self.num_qubits = num_qubits
        self.wormhole_nodes = wormhole_nodes
        self.simulator = UnifiedSpacetimeSimulator(grid_size, lambda_eigen)
        self.circuit = QuantumCircuit(num_qubits)
        self.harmonic_gen = HarmonicGenerator(sample_rate, kappa_j6)
        self.arduino = ArduinoInterface(port)
        self.toe = UnifiedTOE(self.simulator, self.circuit, self.harmonic_gen)
        self.entanglement_cache = EntanglementCache()
        self.weights = np.array([1.0, 1.0, 1.0, 0.1, 0.1, 0.1])
        self.beta = Constants.BETA
        self.k = Constants.K / Constants.DELTA_X
        self.omega = 2 * np.pi / (100 * Constants.DELTA_T)
        self.results_dir = 'results'
        os.makedirs(self.results_dir, exist_ok=True)
        self.gui = FluxGUI(self)
        self.ricci_scalar = np.ones(grid_size, dtype=np.float64)  # Initial Rio placeholder
        logger.info("Flux Capacitor initialized with J^6 coupling")

    def compute_scalar_field(self, r: np.ndarray, t: float) -> float:
        r_6d = np.sqrt(np.sum(self.weights * (r - np.array(self.grid_size) / 2)**2))
        term1 = -r_6d**2 * np.cos(self.k * r_6d - self.omega * t)
        term2 = 2 * r_6d * np.sin(self.k * r_6d - self.omega * t)
        term3 = 2 * np.cos(self.k * r_6d - self.omega * t)
        term4 = 0.1 * np.sin(1e-3 * r_6d)
        phi = -(term1 + term2 + term3 + term4)
        return phi

    def initialize_spin_network(self):
        self.simulator.initialize_tetrahedral_lattice()
        grid_center = np.array(self.grid_size) / 2
        phi = self.compute_scalar_field(grid_center, 0.0)
        self.circuit.apply_rydberg_gates(wormhole_nodes=True, phi=phi)
        chsh_value = self.circuit.compute_chsh_violation()
        if abs(chsh_value - 2.828) > 0.1:
            self.circuit = QuantumCircuit(self.num_qubits)
            self.circuit.apply_rydberg_gates(wormhole_nodes=True, phi=phi)

    def run(self, audio_input: np.ndarray, body_positions: list = None) -> np.ndarray:
        num_samples = int(self.sample_rate * self.duration)
        audio_input = np.tile(audio_input, int(np.ceil(num_samples / audio_input.size)))[:num_samples]
        audio_output = np.zeros(num_samples)
        self.initialize_spin_network()
        grid_center = np.array(self.grid_size) / 2

        # Performance optimization: Generate configuration grid
        # Full grid would be 3^5 * 3^8 * 3 = 1,594,323 combinations
        # Limited to first 2 of each for performance (2 * 2 * 3 = 12 total iterations)
        j6_configs = [
            {'kappa_j6': kj, 'kappa_j6_eff': kje, 'j6_scaling_factor': jsf, 'epsilon': eps, 'resonance_frequency': rf}
            for kj in Constants.J6_PARAM_RANGES['kappa_j6']
            for kje in Constants.J6_PARAM_RANGES['kappa_j6_eff']
            for jsf in Constants.J6_PARAM_RANGES['j6_scaling_factor']
            for eps in Constants.J6_PARAM_RANGES['epsilon']
            for rf in Constants.J6_PARAM_RANGES['resonance_frequency']
        ]
        ctc_configs = [
            {'tau': tau, 'kappa_ctc': kappa, 'r': r, 'n': n, 'm_shift_amplitude': msa,
             'ctc_phase_factor': cpf, 'ctc_wormhole_factor': cwf, 'ctc_amplify_factor': caf}
            for tau in [0.5, 1.0, 1.5]
            for kappa in [0.3, 0.5, 0.7]
            for r in [2.0, 3.0]
            for n in [1.0, 2.0]
            for msa in [2.0, 2.72, 3.5]
            for cpf in [0.1, 0.3]
            for cwf in [0.5, 0.7]
            for caf in [0.2, 0.3]
        ]
        boundary_factors = [0.8, 0.9, 1.0]
        
        results = []
        # Performance optimization: Limit to first 2 configurations for testing
        for j6_config in j6_configs[:2]:
            self.harmonic_gen.kappa_j6 = j6_config['kappa_j6']
            self.harmonic_gen.kappa_j6_eff = j6_config['kappa_j6_eff']
            self.harmonic_gen.j6_scaling_factor = j6_config['j6_scaling_factor']
            self.harmonic_gen.epsilon = j6_config['epsilon']
            self.harmonic_gen.omega_res = j6_config['resonance_frequency'] * 2 * np.pi
            
            for ctc_config in ctc_configs[:2]:
                for boundary_factor in boundary_factors:
                    for i, t in enumerate(np.linspace(0, self.duration, num_samples)):
                        phi = self.compute_scalar_field(grid_center, t)
                        j4 = np.mean(self.simulator.grid)
                        psi = self.simulator.psi
                        psi_abs_sq = np.mean(np.abs(psi)**2)
                        j4_abs = np.abs(j4)
                        graviton_field = self.simulator.sphinx_os.graviton_field if hasattr(self.simulator.sphinx_os, 'graviton_field') else np.zeros(self.grid_size + (6, 6))
                        harmonics = self.harmonic_gen.generate_harmonics(phi, j4, psi, self.ricci_scalar, graviton_field, boundary_factor, body_positions=body_positions)
                        ctc_effect = self.toe.simulator.compute_ctc_term(t, phi, harmonics, ctc_params=ctc_config, body_positions=body_positions)
                        self.circuit.apply_rydberg_gates(wormhole_nodes=True, phi=phi, ctc_feedback=ctc_effect, 
                                                        ctc_params=ctc_config, boundary_factor=boundary_factor,
                                                        psi_abs_sq=psi_abs_sq, j4_abs=j4_abs)
                        audio_output[i] = harmonics
                        self.arduino.send_control_signal(ctc_effect)
                    
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    harmonic_peaks = self.harmonic_gen.analyze_harmonics(audio_output, 
                                                                        os.path.join(self.results_dir, f'j6_harmonics_{timestamp}_tau{ctc_config["tau"]}_kappa{ctc_config["kappa_ctc"]}.png'),
                                                                        body_positions=body_positions)
                    delay = self.harmonic_gen.analyze_delays(audio_output, audio_input)
                    entanglement_entropy = self.simulator.compute_entanglement_entropy()
                    self.entanglement_cache.cache_entropy(entanglement_entropy)
                    chsh_standard = self.circuit.compute_chsh_violation(amplify=False, ctc_params=ctc_config)
                    chsh_amplified = self.circuit.compute_chsh_violation(amplify=True, ctc_params=ctc_config)
                    mabk_standard = self.circuit.compute_mabk_violation(n_qubits=4, amplify=False, ctc_params=ctc_config)
                    mabk_amplified = self.circuit.compute_mabk_violation(n_qubits=4, amplify=True, ctc_params=ctc_config)
                    ghz_standard = self.circuit.compute_ghz_paradox(amplify=False, ctc_params=ctc_config)
                    ghz_amplified = self.circuit.compute_ghz_paradox(amplify=True, ctc_params=ctc_config)
                    j6_stats = self.harmonic_gen.analyze_j6_potential(phi, j4, psi, self.ricci_scalar, graviton_field, boundary_factor)
                    
                    visualize_rio_field(self.ricci_scalar, os.path.join(self.results_dir, f'rio_field_{timestamp}.png'), boundary_factor, body_positions)
                    plot_graviton_field(graviton_field, os.path.join(self.results_dir, f'graviton_field_{timestamp}.png'))
                    
                    harmonic_chi2 = stats.chisquare([1 if abs(p - 880) < 50 or abs(p - 1320) < 50 else 0 for p in harmonic_peaks[:2]], 
                                                   f_exp=[1, 1]).pvalue
                    delay_ttest = stats.ttest_1samp([delay], 0.050).pvalue
                    chsh_valid = abs(chsh_standard - 2.828427) < 0.1
                    mabk_valid = abs(mabk_standard - 5.656854) < 0.2
                    ghz_valid = all(abs(ghz_standard[key] - value) < 0.01 for key, value in 
                                    {'XXX': 1.0, 'XYY': -1.0, 'YXY': -1.0, 'YYX': -1.0}.items())
                    
                    result = {
                        'j6_config': j6_config,
                        'ctc_config': ctc_config,
                        'boundary_factor': boundary_factor,
                        'harmonic_peaks': harmonic_peaks[:5],
                        'delay': delay,
                        'entropy': entanglement_entropy,
                        'chsh_standard': chsh_standard,
                        'chsh_amplified': chsh_amplified,
                        'mabk_standard': mabk_standard,
                        'mabk_amplified': mabk_amplified,
                        'ghz_standard': ghz_standard,
                        'ghz_amplified': ghz_amplified,
                        'j6_stats': j6_stats,
                        'timestamp': timestamp
                    }
                    results.append(result)
                    
                    self.save_results(audio_output, audio_input, result, harmonic_chi2, delay_ttest, chsh_valid, mabk_valid, ghz_valid)
                    logger.info("J^6 config %s, CTC config %s, boundary_factor=%.2f: peaks=%s, delay=%.3f s, entropy=%.3f, CHSH_standard=%.3f, CHSH_amplified=%.3f, MABK_standard=%.3f, MABK_amplified=%.3f, GHZ_standard=%s, GHZ_amplified=%s, harmonic_chi2_p=%.3f, delay_ttest_p=%.3f, chsh_valid=%s, mabk_valid=%s, ghz_valid=%s, rio_mean=%.3f, graviton_trace=%.3f, graviton_nonlinear=%.3e",
                                j6_config, ctc_config, boundary_factor, harmonic_peaks[:5], delay, entanglement_entropy, chsh_standard, chsh_amplified, 
                                mabk_standard, mabk_amplified, ghz_standard, ghz_amplified, harmonic_chi2, delay_ttest, chsh_valid, mabk_valid, ghz_valid, 
                                j6_stats['rio_mean'], j6_stats['graviton_trace'], j6_stats['graviton_nonlinear'])
        
        return audio_output

    def run_three_body_simulation(self, body_positions: list, body_masses: list, velocities: list, duration: float):
        """Simulate three-body motion and compute trajectories."""
        num_samples = int(self.sample_rate * duration)
        audio_output = np.zeros(num_samples)
        trajectories = [[] for _ in range(3)]
        
        for i, t in enumerate(np.linspace(0, duration, num_samples)):
            phi = self.compute_scalar_field(np.array(self.grid_size)/2, t)
            j4 = np.mean(self.simulator.grid)
            psi = self.simulator.psi
            self.ricci_scalar = self.simulator.sphinx_os.ricci_scalar
            graviton_field = self.simulator.sphinx_os.graviton_field
            
            # Update metric, graviton field, curvature, and Nugget field
            self.simulator.sphinx_os.em_fields["metric"], _ = compute_quantum_metric(
                self.simulator.lattice, self.simulator.sphinx_os.nugget_field, self.simulator.psi,
                self.grid_size, self.simulator.sphinx_os.em_fields["J4"], self.simulator.psi,
                body_positions, body_masses
            )
            self.simulator.sphinx_os.graviton_field, _ = evolve_graviton_field(
                self.simulator.sphinx_os.graviton_field, self.grid_size, self.simulator.sphinx_os.deltas,
                self.simulator.sphinx_os.dt, self.simulator.sphinx_os.nugget_field, self.ricci_scalar,
                self.simulator.psi, self.simulator.sphinx_os.em_fields["J4"], body_positions, body_masses
            )
            _, self.ricci_scalar = compute_curvature(
                self.simulator.sphinx_os.em_fields["metric"], 
                np.linalg.inv(self.simulator.sphinx_os.em_fields["metric"]),
                self.grid_size, self.simulator.sphinx_os.deltas[1], self.simulator.sphinx_os.nugget_field,
                body_positions
            )
            self.simulator.sphinx_os.nugget_field, _ = evolve_nugget_field(
                self.simulator.sphinx_os.nugget_field, self.grid_size, self.simulator.sphinx_os.deltas,
                self.simulator.sphinx_os.dt, self.simulator.sphinx_os.graviton_field,
                self.ricci_scalar, self.simulator.psi, j4, body_positions, body_masses
            )
            
            # Compute forces and update positions
            forces = []
            G = 6.67430e-11
            for i in range(3):
                force = np.zeros(3)
                for j in range(3):
                    if i != j:
                        r_ij = body_positions[j] - body_positions[i]
                        dist = np.sqrt(np.sum(r_ij**2) + 1e-15)
                        force += G * body_masses[i] * body_masses[j] * r_ij / dist**3
                forces.append(force)
            
            # Update positions and velocities
            dt = duration / num_samples
            for i in range(3):
                velocities[i] += forces[i] / body_masses[i] * dt
                body_positions[i] += velocities[i] * dt
                trajectories[i].append(body_positions[i].copy())
            
            # Generate harmonics
            harmonics = self.harmonic_gen.generate_harmonics(phi, j4, psi, self.ricci_scalar, graviton_field, 
                                                            body_positions=body_positions)
            audio_output[i] = harmonics
        
        # Save trajectories
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        np.save(os.path.join(self.results_dir, f'trajectories_{timestamp}.npy'), np.array(trajectories))
        dist_sum = compute_body_distance_sum(body_positions)
        logger.info("Three-body trajectories saved to %s, dist_sum=%.6f", 
                    os.path.join(self.results_dir, f'trajectories_{timestamp}.npy'), dist_sum)
        return audio_output

    def save_results(self, audio_output: np.ndarray, audio_input: np.ndarray, result: dict, 
                     harmonic_chi2: float, delay_ttest: float, chsh_valid: bool, mabk_valid: bool, ghz_valid: bool) -> None:
        timestamp = result['timestamp']
        j6_config = result['j6_config']
        ctc_config = result['ctc_config']
        boundary_factor = result['boundary_factor']
        audio_path = os.path.join(self.results_dir, f'audio_output_{timestamp}.npy')
        np.save(audio_path, audio_output)
        fft_path = os.path.join(self.results_dir, f'fft_plot_{timestamp}.png')
        plot_fft(audio_output, self.sample_rate, fft_path)
        metadata_path = os.path.join(self.results_dir, f'metadata_{timestamp}_tau{ctc_config["tau"]}_kappa{ctc_config["kappa_ctc"]}.txt')
        with open(metadata_path, 'w') as f:
            f.write(f"Timestamp: {timestamp}\nGrid Size: {self.grid_size}\nLambda Eigen: {self.lambda_eigen}\n"
                    f"J^6 Config: {j6_config}\nCTC Config: {ctc_config}\nBoundary Factor: {boundary_factor}\n"
                    f"Sample Rate: {self.sample_rate}\nDuration: {self.duration}\nNumber of Qubits: {self.num_qubits}\n"
                    f"Wormhole Nodes: {self.wormhole_nodes}\nEntanglement Entropy: {result['entropy']:.6f}\n"
                    f"CHSH Violation (Standard): {result['chsh_standard']:.6f}\n"
                    f"CHSH Violation (Amplified): {result['chsh_amplified']:.6f}\n"
                    f"MABK Violation (Standard, n=4): {result['mabk_standard']:.6f}\n"
                    f"MABK Violation (Amplified, n=4): {result['mabk_amplified']:.6f}\n"
                    f"GHZ Paradox (Standard, n=3): {result['ghz_standard']}\n"
                    f"GHZ Paradox (Amplified, n=3): {result['ghz_amplified']}\n"
                    f"J^6 Stats: {result['j6_stats']}\n"
                    f"Delay: {result['delay']:.6f} s\n"
                    f"Harmonic Chi-squared p-value: {harmonic_chi2:.6f}\n"
                    f"Delay t-test p-value: {delay_ttest:.6f}\n"
                    f"CHSH Valid: {chsh_valid}\n"
                    f"MABK Valid: {mabk_valid}\n"
                    f"GHZ Valid: {ghz_valid}\n")
