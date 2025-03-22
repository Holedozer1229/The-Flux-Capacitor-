import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.signal import resample
from scipy.linalg import expm
import sounddevice as sd
import serial
from mpl_toolkits.mplot3d import Axes3D
import warnings
import time
import csv
import logging

warnings.filterwarnings('ignore')

# Constants
RS = 2.0  # meters

# Configuration dictionary with units
CONFIG = {
    "swarm_size": 5,                      # Dimensionless
    "max_iterations": 200,                # Dimensionless
    "resolution": 20,                     # Dimensionless
    "time_delay_steps": 3,                # Dimensionless
    "ctc_feedback_factor": 1.6180339887,  # Dimensionless
    "entanglement_factor": 0.2,           # Dimensionless
    "charge": 1.0,                        # Coulombs (C)
    "em_strength": 3.0,                   # Volts per meter (V/m)
    "nodes": 16,                          # Dimensionless
    "alpha_em": 1/137,                    # Dimensionless
    "phi_N_evolution_factor": 1e-6,       # Dimensionless (assuming F_squared is normalized)
    "j4_coupling_factor": 1e-18,          # m^8 s^4 C^-4 (to make j4_effect dimensionless)
    "steps": 5                            # Dimensionless
}

# Global variables
TARGET_PHYSICAL_STATE = int(time.time() * 1000)
START_TIME = time.perf_counter_ns() / 1e9
KNOWN_STATE = int(START_TIME * 1000) % 2**32

# Logging setup
logging.basicConfig(filename='flux_capacitor_ctc.log', level=logging.INFO, 
                    format='%(asctime)s.%(msecs)03d - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger("FluxCapacitorCTC")

def repeating_curve(index):
    """Returns 1 if index is even, 0 if odd."""
    return 1 if index % 2 == 0 else 0

# Supporting Classes
class SpinNetwork:
    """Handles spin network computations."""
    def __init__(self, nodes=CONFIG["nodes"]):
        self.nodes = nodes
        self.edges = [(i, (i + 1) % nodes) for i in range(nodes)]
        self.state = np.ones(nodes, dtype=complex) / np.sqrt(nodes)

    def evolve(self, H, dt):
        """Evolves the spin network state."""
        self.state = expm(-1j * H * dt) @ self.state

    def get_adjacency_matrix(self):
        """Returns the adjacency matrix of the spin network."""
        A = np.zeros((self.nodes, self.nodes))
        for i, j in self.edges:
            A[i, j] = A[j, i] = 1
        return A

class CTCTetrahedralField:
    """Manages tetrahedral field computations."""
    def __init__(self, resolution=CONFIG["resolution"]):
        self.resolution = resolution
        self.coordinates = self._generate_tetrahedral_coordinates()
        self.H = self._build_hamiltonian()

    def _generate_tetrahedral_coordinates(self):
        """Generates tetrahedral coordinates."""
        coords = np.zeros((self.resolution, 4))
        t = np.linspace(0, 2 * np.pi, self.resolution)
        coords[:, 0] = np.cos(t) * np.sin(t)
        coords[:, 1] = np.sin(t) * np.sin(t)
        coords[:, 2] = np.cos(t)
        coords[:, 3] = t / (2 * np.pi)
        return coords

    def _build_hamiltonian(self):
        """Builds the Hamiltonian for the tetrahedral field."""
        H = np.zeros((self.resolution, self.resolution), dtype=complex)
        for i in range(self.resolution):
            for j in range(i + 1, self.resolution):
                Δx = self.coordinates[j] - self.coordinates[i]
                distance = np.linalg.norm(Δx)
                if distance > 0:
                    H[i, j] = H[j, i] = 1j / (distance + 1e-10)
        np.fill_diagonal(H, -1j * np.linalg.norm(self.coordinates[:, :3], axis=1))
        return H

    def propagate(self, ψ0, τ):
        """Propagates the quantum state."""
        return expm(-1j * self.H * τ) @ ψ0

# Main Simulator Class
class UnifiedSpacetimeSimulator:
    """Simulates a unified spacetime system with quantum fields and particle dynamics."""
    def __init__(self, resolution=CONFIG["resolution"], lambda_=2.72, kappa=0.1, charge_density=1e-12, serial_port='COM3', baud_rate=9600):
        # Simulation parameters with units
        self.resolution = resolution          # Dimensionless
        self.lambda_ = lambda_                # Meters (m)
        self.kappa = kappa                    # Dimensionless
        self.charge_density = charge_density  # C/m^3
        self.beta = 0.1                       # Dimensionless
        self.g_strong = 1.0                   # Dimensionless coupling constant
        self.g_weak = 0.65                    # Dimensionless coupling constant
        self.mass_e = 9.11e-31                # kg
        self.mass_q = 2.3e-30                 # kg
        self.c = 3e8                          # m/s
        self.hbar = 1.0545718e-34             # J s
        self.eps_0 = 8.854187817e-12          # F/m (permittivity of free space)
        self.G = 6.67430e-11                  # m^3 kg^-1 s^-2 (gravitational constant)

        self.dx = 4.0 / (self.resolution - 1)  # m
        self.dt = 1e-3                         # s
        self.time = 0.0                        # s

        # Physical parameters
        self.schumann_freqs = [7.83, 14.3, 20.8, 27.3, 33.8]  # Hz
        self.flux_freq = 0.00083                              # Hz
        self.schumann_amplitudes = [1.0, 0.5, 0.33, 0.25, 0.2]  # Dimensionless
        self.pythagorean_ratios = [1.0, 2.0, 3/2, 4/3]          # Dimensionless
        self.wormhole_nodes = [0]

        # Serial connection
        try:
            self.arduino = serial.Serial(serial_port, baud_rate, timeout=1)
            time.sleep(2)
            print(f"Connected to Arduino on {serial_port}")
        except serial.SerialException as e:
            print(f"Error connecting to Arduino: {e}")
            self.arduino = None

        # Initialize components
        self.spin_network = SpinNetwork()
        self.tetrahedral_field = CTCTetrahedralField()
        self.bit_states = np.array([repeating_curve(i) for i in range(self.resolution)], dtype=int)
        self.temporal_entanglement = np.zeros(self.resolution)
        self.quantum_state = np.ones(self.resolution, dtype=complex) / np.sqrt(self.resolution)
        self.history = []

        # Initialize fields
        self.fabric, self.edges = self.generate_spacetime_fabric()
        self.quantum = self.init_quantum_fields()
        self.em = self.init_em_fields()
        self.iteration = 0
        self.ctc_influence = 0.0
        self.delta_g_tphi = np.zeros(self.resolution)
        self.metric = self.compute_metric()
        self.christoffel = self.compute_christoffel_symbols_analytical()
        self.strong = self.init_strong_fields()
        self.weak = self.init_weak_fields()
        self.gw = self.init_gravitational_waves()
        self.stress_energy = np.zeros((resolution, 4, 4), dtype=np.float32)
        self.particles = []

        logger.info(f"Init, Time {int(self.time * 1e9)}: Bit States = {self.bit_states.tolist()}")

    def generate_spacetime_fabric(self):
        """Generates the spacetime fabric with vertices and edges."""
        scale = 2 / (3 * np.sqrt(2))  # Dimensionless
        vertices = []
        WORMHOLE_VECTOR = np.array([0.33333333326, 0.33333333326, 0.33333333326, 0.33333333326], dtype=np.float32)
        
        for i in range(self.resolution):
            if i in self.wormhole_nodes:
                vertices.append(WORMHOLE_VECTOR)
            else:
                theta = 2 * np.pi * i / self.resolution
                phi = np.pi * i / self.resolution
                x = scale * np.sin(phi) * np.cos(theta)
                y = scale * np.sin(phi) * np.sin(theta)
                z = scale * np.cos(phi)
                t = i / self.resolution
                vertices.append(np.array([x, y, z, t], dtype=np.float32))
        
        edges = [(i, (i + 1) % self.resolution) for i in range(self.resolution)]
        spins = np.random.choice([0.5, 1.0], len(edges))
        fabric = np.array(vertices, dtype=np.float32)
        return fabric, list(zip(edges, spins))

    def init_quantum_fields(self):
        """Initializes quantum fields with dimensionless phi_N."""
        r = np.linalg.norm(self.fabric[:, :3], axis=1)  # m
        phi_N = (1 / self.kappa) * (1 - np.exp(- (r / self.lambda_)**2))  # Dimensionless
        return {
            'spinor': np.random.normal(0, 1e-3, (self.resolution, 4)).astype(np.complex64),
            'quark_spinors': np.random.normal(0, 1e-3, (self.resolution, 2, 3, 4)).astype(np.complex64),
            'phi_N': phi_N.astype(np.float32)
        }

    def evolve_phi_N(self):
        """Evolves the phi_N field."""
        F_munu = self.em['F_munu']
        J4 = self.em['J4']
        F_squared = np.zeros(self.resolution, dtype=np.float32)
        for i in range(self.resolution):
            for mu in range(4):
                for nu in range(4):
                    F_squared[i] += F_munu[i, mu, nu] * F_munu[i, mu, nu]
            F_squared[i] += CONFIG["j4_coupling_factor"] * J4[i]
        delta_phi_N = CONFIG["phi_N_evolution_factor"] * F_squared * self.dt
        self.quantum['phi_N'] += delta_phi_N
        self.quantum['phi_N'] = np.clip(self.quantum['phi_N'], 0, 10.0)
        return self.quantum['phi_N']

    def compute_metric(self):
        """Computes the spacetime metric with dimensionless terms."""
        metric = np.tile(np.eye(4, dtype=np.float32)[np.newaxis, :, :], (self.resolution, 1, 1))
        r = np.linalg.norm(self.fabric[:, :3], axis=1)
        r = np.where(r < 1e-6, 1e-6, r)  # m
        phi_N = self.quantum['phi_N'][:, np.newaxis, np.newaxis]  # Dimensionless
        J4 = self.em['J4'][:, np.newaxis, np.newaxis]  # (C/(m^2 s))^4

        mass_term = 2 * self.G * 1.989e30 / (self.c**2 * r)  # Dimensionless
        charge_term = (self.G * self.em['charge']**2) / (4 * np.pi * self.eps_0 * self.c**4 * r**2)  # Dimensionless

        for i in range(self.resolution):
            j4_term = CONFIG["j4_coupling_factor"] * J4[i]  # Dimensionless
            metric[i, 0, 0] = -(1 - mass_term[i] + charge_term[i] + j4_term) * (1 + self.kappa * phi_N[i])
            metric[i, 1, 1] = 1 / (1 - mass_term[i] + charge_term[i] + j4_term)
            metric[i, 2, 2] = r[i]**2 * (1 + self.kappa * phi_N[i] + j4_term)
            metric[i, 3, 3] = metric[i, 2, 2] * np.sin(self.fabric[i, 2])**2
            metric[i, 0, 3] = 0.27 * self.lambda_ * np.exp(-r[i] / self.lambda_) + self.delta_g_tphi[i]
            metric[i, 3, 0] = metric[i, 0, 3]
        return metric

    def compute_inverse_metric(self):
        """Computes the inverse metric."""
        g_inv = np.zeros_like(self.metric, dtype=np.float32)
        for i in range(self.resolution):
            try:
                g_inv[i] = np.linalg.inv(self.metric[i])
            except np.linalg.LinAlgError:
                g_inv[i] = np.eye(4, dtype=np.float32)
        return g_inv

    def compute_christoffel_symbols_analytical(self, h=1e-4):
        """Computes Christoffel symbols analytically."""
        christoffel = np.zeros((self.resolution, 4, 4, 4), dtype=np.float32)
        for i in range(self.resolution):
            if np.linalg.norm(self.fabric[i, :3]) < 0.1:
                continue
            for mu in range(4):
                for nu in range(4):
                    for sigma in range(4):
                        dg_mu = (self.metric[(i+1)%self.resolution, mu, nu] - 
                                 self.metric[(i-1)%self.resolution, mu, nu]) / (2*h + 1e-6)
                        dg_nu = (self.metric[(i+1)%self.resolution, nu, mu] - 
                                 self.metric[(i-1)%self.resolution, nu, mu]) / (2*h + 1e-6)
                        dg_sigma = (self.metric[(i+1)%self.resolution, mu, nu] - 
                                    self.metric[(i-1)%self.resolution, mu, nu]) / (2*h + 1e-6)
                        christoffel[i, mu, nu, sigma] = 0.5 * (dg_mu + dg_nu - dg_sigma)
        return christoffel

    def compute_maxwell_equations(self):
        """Solves Maxwell's equations with J4 coupling."""
        F_munu = self.em['F_munu']
        J_nu = self.em['J']
        J4 = self.em['J4']
        g_inv = self.compute_inverse_metric()
        
        F_upper = np.zeros((self.resolution, 4, 4), dtype=np.float32)
        for i in range(self.resolution):
            for mu in range(4):
                for nu in range(4):
                    for alpha in range(4):
                        for beta in range(4):
                            F_upper[i, mu, nu] += g_inv[i, mu, alpha] * g_inv[i, nu, beta] * F_munu[i, alpha, beta]
        
        div_F = np.zeros((self.resolution, 4), dtype=np.float32)
        for i in range(self.resolution):
            if np.linalg.norm(self.fabric[i, :3]) < 0.1:
                continue
            for nu in range(4):
                for mu in range(4):
                    dF_mu = (F_upper[(i+1)%self.resolution, mu, nu] - 
                             F_upper[(i-1)%self.resolution, mu, nu]) / (2 * self.dx)
                    for lam in range(4):
                        dF_mu += self.christoffel[i, mu, mu, lam] * F_upper[i, lam, nu]
                        dF_mu += self.christoffel[i, mu, nu, lam] * F_upper[i, mu, lam]
                    div_F[i, nu] += dF_mu

        for i in range(self.resolution):
            for nu in range(4):
                correction = (4 * np.pi * self.lambda_ * J_nu[i, nu] - div_F[i, nu] + 
                            CONFIG["j4_coupling_factor"] * J4[i]) * self.dt
                for mu in range(4):
                    F_munu[i, mu, nu] += correction / 4
                    F_munu[i, nu, mu] = -F_munu[i, mu, nu]

        self.em['F_munu'] = F_munu
        return div_F

    def compute_delta_g_tphi(self):
        """Computes the off-diagonal metric perturbation."""
        F_munu = self.em['F_munu']
        J4 = self.em['J4']
        F_squared = np.zeros(self.resolution, dtype=np.float32)
        g_inv = self.compute_inverse_metric()
        for i in range(self.resolution):
            for mu in range(4):
                for nu in range(4):
                    F_upper = 0
                    for alpha in range(4):
                        for beta in range(4):
                            F_upper += g_inv[i, mu, alpha] * g_inv[i, nu, beta] * F_munu[i, alpha, beta]
                    F_squared[i] += F_upper * F_munu[i, mu, nu]
            F_squared[i] += CONFIG["j4_coupling_factor"] * J4[i]
        self.delta_g_tphi += CONFIG["alpha_em"] * F_squared * self.dt
        return self.delta_g_tphi

    def init_em_fields(self):
        """Initializes electromagnetic fields."""
        em = {
            'A_mu': np.zeros((self.resolution, 4), dtype=np.float32),  # V for A[0], V s / m for A[1:4]
            'F_munu': np.zeros((self.resolution, 4, 4), dtype=np.float32),  # V/m for electric, T for magnetic
            'charge': CONFIG["charge"],  # C
            'J': np.zeros((self.resolution, 4), dtype=np.float32),  # C/(m^2 s)
            'J4': np.zeros(self.resolution, dtype=np.float32)  # (C/(m^2 s))^4
        }
        em['J'][:, 0] = self.charge_density * self.c  # C/(m^2 s)
        em['J4'] = np.power(np.linalg.norm(em['J'], axis=1), 4)  # (C/(m^2 s))^4
        return em

    def init_strong_fields(self):
        """Initializes strong interaction fields."""
        return {
            'A_mu': np.zeros((self.resolution, 8, 4), dtype=np.float32),
            'F_munu': np.zeros((self.resolution, 8, 4, 4), dtype=np.float32)
        }

    def init_weak_fields(self):
        """Initializes weak interaction fields."""
        return {
            'W_mu': np.zeros((self.resolution, 3, 4), dtype=np.float32),
            'W_munu': np.zeros((self.resolution, 3, 4, 4), dtype=np.float32),
            'higgs': np.ones(self.resolution, dtype=np.float32) * 246e9 / self.c
        }

    def init_gravitational_waves(self):
        """Initializes gravitational wave perturbations."""
        t = self.fabric[:, 3]
        f_schumann = 7.83
        return {
            'plus': 1e-6 * np.sin(2 * np.pi * f_schumann * t),
            'cross': 1e-6 * np.cos(2 * np.pi * f_schumann * t)
        }

    def compute_vector_potential(self, iteration):
        """Computes the vector potential with correct units."""
        A = np.zeros((self.resolution, 4))  # A[0] in V, A[1:4] in V s / m
        r = np.linalg.norm(self.fabric[:, :3], axis=1)  # m
        theta = self.fabric[:, 2]  # radians
        J4 = self.em['J4']  # (C/(m^2 s))^4
        load_factor = (time.perf_counter_ns() / 1e9 - START_TIME) / 5  # Dimensionless

        for i in range(self.resolution):
            j4_effect = CONFIG["j4_coupling_factor"] * J4[i]  # Dimensionless
            A[i, 0] = - CONFIG["charge"] / (4 * np.pi * self.eps_0 * r[i]) * (1 + np.sin(iteration * 0.2) * load_factor + j4_effect)  # V
            A[i, 3] = (CONFIG["em_strength"] / self.c) * r[i] * np.sin(theta[i]) * (1 + load_factor + j4_effect)  # V s / m
        return A

    def quantum_walk(self, iteration, current_time):
        """Performs a quantum walk on the system."""
        A_mu = self.compute_vector_potential(iteration)
        self.em['A_mu'] = A_mu
        prob = np.abs(self.quantum_state)**2
        adj_matrix = self.spin_network.get_adjacency_matrix()
        self.spin_network.evolve(adj_matrix, 2 * np.pi / self.resolution)
        J4 = self.em['J4']
        for idx in range(self.resolution):
            expected_state = repeating_curve(idx + iteration)
            self.bit_states[idx] = expected_state
            window = prob[max(0, idx - CONFIG["time_delay_steps"]):idx + 1]
            self.temporal_entanglement[idx] = CONFIG["entanglement_factor"] * np.mean(window) if window.size > 0 else 0
            em_perturbation = A_mu[idx, 0] * CONFIG["em_strength"]
            j4_perturbation = CONFIG["j4_coupling_factor"] * J4[idx]
            if np.random.random() < abs(em_perturbation + j4_perturbation) * self.temporal_entanglement[idx]:
                self.bit_states[idx] = 1 - self.bit_states[idx]
        self.quantum_state = self.tetrahedral_field.propagate(self.quantum_state, 2 * np.pi / self.resolution)
        self.history.append((int(current_time * 1e9), self.bit_states.copy()))
        em_effect = np.mean(np.abs(A_mu[:, 0]))
        logger.info(f"Iteration {iteration}, Time {int(current_time * 1e9)}: Bit States = {self.bit_states.tolist()}, "
                    f"Entanglement = {self.temporal_entanglement[0]:.4f}, EM Effect = {em_effect:.6f}")

    def compute_fitness(self, state, temporal_pos):
        """Computes the fitness of a state."""
        current_time = time.perf_counter_ns() / 1e9
        delta_time = current_time - temporal_pos
        base_fitness = abs(state - KNOWN_STATE)
        ctc_influence = 0
        if self.iteration >= CONFIG["time_delay_steps"] and len(self.history) >= CONFIG["time_delay_steps"]:
            past_states = [h[1] for h in self.history[-CONFIG["time_delay_steps"]:]]
            ctc_influence = np.mean([s[0] for s in past_states]) * CONFIG["ctc_feedback_factor"]
            self.ctc_influence = 1.6667 if self.iteration % 2 == 0 else 3.3333
        j4_effect = CONFIG["j4_coupling_factor"] * np.mean(self.em['J4'])
        fitness = base_fitness + ctc_influence + j4_effect
        return fitness, delta_time, ctc_influence

    def generate_fourier_borel_signal(self, t, num_terms=10):
        """Generates a Fourier-Borel signal."""
        f_signal = np.zeros_like(t, dtype=np.float32)
        for n in range(1, num_terms + 1):
            k = 2 * n - 1
            x = 2 * np.pi * self.flux_freq * t
            f_signal += (1 / k) * np.sin(k * x)
        f_signal *= (4 / np.pi)
        return f_signal

    def generate_flux_signal(self, duration=10.0, sample_rate=44100, num_fourier_terms=10):
        """Generates the flux capacitor signal."""
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        flux_signal = np.zeros_like(t, dtype=np.float32)
        fourier_borel = self.generate_fourier_borel_signal(t, num_terms=num_fourier_terms)

        for i in range(self.resolution):
            A_mu_norm = np.linalg.norm(self.em['A_mu'][i])
            gw_plus = self.gw['plus'][i]
            gw_cross = self.gw['cross'][i]
            base_flux = A_mu_norm + gw_plus + gw_cross

            schumann_mod = sum(A * np.cos(2 * np.pi * f * t) for f, A in zip(self.schumann_freqs, self.schumann_amplitudes))
            flux_mod = np.cos(2 * np.pi * self.flux_freq * t)
            modulated_flux = base_flux * schumann_mod * flux_mod

            j4_effect = CONFIG["j4_coupling_factor"] * self.em['J4'][i]
            modulated_flux += j4_effect * np.sin(2 * np.pi * self.flux_freq * t)

            harmonic_scale = self.pythagorean_ratios[i % len(self.pythagorean_ratios)]
            scaled_flux = modulated_flux * harmonic_scale + fourier_borel

            if self.ctc_influence > 0:
                delay_samples = int(self.ctc_influence * sample_rate)
                if len(flux_signal) > delay_samples:
                    scaled_flux += 0.1 * flux_signal[:-delay_samples]

            flux_signal += np.power(np.abs(scaled_flux + j4_effect), 4) * np.sign(scaled_flux + j4_effect)

        flux_signal = np.clip(flux_signal, -1.0, 1.0)
        return flux_signal

    def activate_flux_capacitor(self, signal, sample_rate=44100):
        """Activates the flux capacitor with audio and serial output."""
        print("Activating Flux Capacitor with CTC enhancement and J^4 coupling!")
        sd.play(signal, sample_rate)
        
        if self.arduino:
            try:
                scaled_signal = ((signal + 1) * 127.5).astype(int)
                start_time = time.perf_counter_ns() / 1e9
                for idx, value in enumerate(scaled_signal):
                    self.arduino.write(bytes([value]))
                    current_time = time.perf_counter_ns() / 1e9
                    if self.arduino.in_waiting > 0:
                        feedback = self.arduino.readline().decode().strip()
                        try:
                            hall_value = float(feedback)
                            self.em['J'][:, 0] += hall_value * 1e-6
                            self.em['J4'] = np.power(np.linalg.norm(self.em['J'], axis=1), 4)
                        except ValueError:
                            pass
                    
                    state = int(np.sum(self.bit_states * (2 ** np.arange(self.resolution))))
                    fitness, delta_t, ctc_influence = self.compute_fitness(state, start_time)
                    em_effect = np.mean(self.em['J4']) * 1e-20
                    logger.info(f"Iteration {self.iteration}, Time {int(current_time * 1e9)}: "
                                f"Bit States = {self.bit_states.tolist()}, Entanglement = {self.temporal_entanglement[0]:.4f}, "
                                f"EM Effect = {em_effect:.6f}, State = {state}, Fitness = {fitness:.2f}, "
                                f"DeltaT = {delta_t:.6f}, CTC Influence = {ctc_influence:.4f}")
                    
                    time.sleep(1 / sample_rate)
            except serial.SerialException as e:
                print(f"Error communicating with Arduino: {e}")
        
        sd.wait()

    def add_particle(self, position, velocity, charge):
        """Adds a charged particle to the simulation."""
        particle = {
            'position': np.array(position, dtype=np.float32),
            'velocity': np.array(velocity, dtype=np.float32),
            'charge': charge,
            'path': [position[:3].copy()]
        }
        self.particles.append(particle)

    def move_charged_particles(self, dt):
        """Updates particle positions using equations of motion."""
        for p in self.particles:
            state = np.concatenate([p['position'][:3], p['velocity'][:3]])
            t_range = np.linspace(0, dt, 2)
            trajectory = odeint(self.equations_of_motion, state, t_range)
            p['position'][:3] = trajectory[-1, :3]
            p['velocity'][:3] = trajectory[-1, 3:]
            p['path'].append(p['position'][:3].copy())

    def equations_of_motion(self, y, t):
        """Computes the equations of motion for charged particles."""
        x, y_coord, z, vx, vy, vz = y
        i = self.find_closest_fabric_point(x, y_coord, z)
        v2 = (vx**2 + vy**2 + vz**2) / self.c**2
        gamma = 1.0 / np.sqrt(1 - v2) if v2 < 1 else 1.0
        u = np.array([gamma, gamma*vx/self.c, gamma*vy/self.c, gamma*vz/self.c], dtype=np.float32)
        geo_accel = np.zeros(3, dtype=np.float32)
        for spatial_dim in range(3):
            Γ = self.christoffel[i, spatial_dim+1, :, :]
            geo_accel[spatial_dim] = -np.einsum('ij,i,j', Γ, u, u) * self.c**2
        em_force = self.em['charge'] * np.einsum('ij,j', self.em['F_munu'][i, 1:4, :], u)[:3]
        strong_force = np.zeros(3, dtype=np.float32)
        for a in range(8):
            strong_force += self.g_strong * np.einsum('ij,j', self.strong['F_munu'][i, a, 1:4, :], u)[:3]
        weak_force = np.zeros(3, dtype=np.float32)
        for a in range(3):
            weak_force += self.g_weak * np.einsum('ij,j', self.weak['W_munu'][i, a, 1:4, :], u)[:3]
        j4_force = CONFIG["j4_coupling_factor"] * self.em['J4'][i] * u[:3]
        coordinate_accel = (geo_accel + em_force + strong_force + weak_force + j4_force) / gamma**2
        coordinate_accel = np.clip(coordinate_accel, -1e10, 1e10)
        return [vx, vy, vz, coordinate_accel[0], coordinate_accel[1], coordinate_accel[2]]

    def find_closest_fabric_point(self, x, y, z):
        """Finds the closest point in the spacetime fabric."""
        r_point = np.sqrt(x**2 + y**2 + z**2)
        i = np.argmin(np.abs(np.linalg.norm(self.fabric[:, :3], axis=1) - r_point))
        return i

    def evolve_quantum_field_rk4(self):
        """Evolves quantum fields using RK4 method."""
        dt = self.dt
        for i in range(self.resolution):
            t = self.time
            psi = self.quantum['spinor'][i]
            k1 = -1j * self.dirac_hamiltonian(psi, i, t) / self.hbar
            k2 = -1j * self.dirac_hamiltonian(psi + 0.5 * dt * k1, i, t + 0.5 * dt) / self.hbar
            k3 = -1j * self.dirac_hamiltonian(psi + 0.5 * dt * k2, i, t + 0.5 * dt) / self.hbar
            k4 = -1j * self.dirac_hamiltonian(psi + dt * k3, i, t + dt) / self.hbar
            self.quantum['spinor'][i] += (k1 + 2 * k2 + 2 * k3 + k4) * (dt / 6)
            for flavor in range(2):
                for color in range(3):
                    psi_q = self.quantum['quark_spinors'][i, flavor, color]
                    k1_q = -1j * self.dirac_hamiltonian(psi_q, i, t, quark=True, flavor=flavor, color=color) / self.hbar
                    k2_q = -1j * self.dirac_hamiltonian(psi_q + 0.5 * dt * k1_q, i, t + 0.5 * dt, quark=True, flavor=flavor, color=color) / self.hbar
                    k3_q = -1j * self.dirac_hamiltonian(psi_q + 0.5 * dt * k2_q, i, t + 0.5 * dt, quark=True, flavor=flavor, color=color) / self.hbar
                    k4_q = -1j * self.dirac_hamiltonian(psi_q + dt * k3_q, i, t + dt, quark=True, flavor=flavor, color=color) / self.hbar
                    self.quantum['quark_spinors'][i, flavor, color] += (k1_q + 2 * k2_q + 2 * k3_q + k4_q) * (dt / 6)
        self.time += dt

    def dirac_hamiltonian(self, psi, i, t, quark=False, flavor=None, color=None):
        """Computes the Dirac Hamiltonian."""
        gamma_mu = self.dirac_gamma_matrices(self.metric[i])
        H_psi = np.zeros(4, dtype=np.complex64)
        mass = self.mass_q if quark else self.mass_e
        for mu in range(1, 4):
            D_mu_psi = self.covariant_derivative(psi, i, mu)
            H_psi -= 1j * self.c * gamma_mu[0] @ gamma_mu[mu] @ D_mu_psi
        H_psi += (mass * self.c**2 / self.hbar) * gamma_mu[0] @ psi
        H_psi -= 1j * self.em['charge'] * sum(self.em['A_mu'][i, mu] * gamma_mu[mu] @ psi for mu in range(4))
        H_psi += self.schumann_potential(t) * psi
        j4_term = CONFIG["j4_coupling_factor"] * self.em['J4'][i] * psi
        H_psi += j4_term
        if quark and flavor is not None and color is not None:
            T_a = self.gell_mann_matrices()
            psi_full_color = self.quantum['quark_spinors'][i, flavor]
            strong_term = np.zeros(4, dtype=np.complex64)
            for a in range(8):
                for mu in range(4):
                    A_mu_a = self.strong['A_mu'][i, a, mu]
                    color_contribution = T_a[a] @ psi_full_color
                    strong_term += A_mu_a * color_contribution[color]
            H_psi += -1j * self.g_strong * strong_term
        return H_psi

    def covariant_derivative(self, psi, i, mu):
        """Computes the covariant derivative of a spinor."""
        if mu != 0:
            psi_plus = self.quantum['spinor'][(i + 1) % self.resolution] if i + 1 < self.resolution else psi
            psi_minus = self.quantum['spinor'][(i - 1) % self.resolution] if i - 1 >= 0 else psi
            partial_psi = (psi_plus - psi_minus) / (2 * self.dx)
            harmonic_scale = self.pythagorean_ratios[mu % len(self.pythagorean_ratios)]
            partial_psi *= harmonic_scale
        else:
            partial_psi = 0
        Gamma_mu = self.christoffel[i, mu]
        connection_term = np.dot(Gamma_mu, psi)
        return partial_psi + connection_term

    def dirac_gamma_matrices(self, g_mu_nu):
        """Generates Dirac gamma matrices in curved spacetime."""
        gamma_flat = [
            np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]], dtype=np.complex64),
            np.array([[0, 0, 0, 1], [0, 0, 1, 0], [0, -1, 0, 0], [-1, 0, 0, 0]], dtype=np.complex64),
            np.array([[0, 0, 0, -1j], [0, 0, 1j, 0], [0, 1j, 0, 0], [-1j, 0, 0, 0]], dtype=np.complex64),
            np.array([[0, 0, 1, 0], [0, 0, 0, -1], [-1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.complex64)
        ]
        e_a_mu = np.zeros((4, 4), dtype=np.float64)
        for mu in range(4):
            e_a_mu[mu, mu] = np.sqrt(np.abs(g_mu_nu[mu, mu]))
        e_mu_a = np.linalg.inv(e_a_mu)
        gamma_mu = [sum(e_mu_a[mu, a] * gamma_flat[a] for a in range(4)) for mu in range(4)]
        return gamma_mu

    def gell_mann_matrices(self):
        """Returns Gell-Mann matrices for SU(3) symmetry."""
        return [
            np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=np.complex64),
            np.array([[0, -1j, 0], [1j, 0, 0], [0, 0, 0]], dtype=np.complex64),
            np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]], dtype=np.complex64),
            np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]], dtype=np.complex64),
            np.array([[0, 0, -1j], [0, 0, 0], [1j, 0, 0]], dtype=np.complex64),
            np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=np.complex64),
            np.array([[0, 0, 0], [0, 0, -1j], [0, 1j, 0]], dtype=np.complex64),
            np.array([[1, 0, 0], [0, 1, 0], [0, 0, -2]], dtype=np.complex64) / np.sqrt(3)
        ]

    def schumann_potential(self, t):
        """Computes the Schumann resonance potential."""
        V_0 = 1e-6
        V_t = 0
        for fn, An in zip(self.schumann_freqs, self.schumann_amplitudes):
            V_t += V_0 * An * np.cos(2 * np.pi * fn * t)
        return V_t

    def total_stress_energy(self):
        """Computes the total stress-energy tensor."""
        r = np.linalg.norm(self.fabric[:, :3], axis=1)
        r = np.where(r < 1e-6, 1e-6, r)
        phi_N = self.quantum['phi_N']
        J4 = self.em['J4']
        T = np.zeros((self.resolution, 4, 4), dtype=np.float32)
        T[:, 0, 0] = -phi_N / self.c**2  # s^-2
        T_EM = np.einsum('...ij,...kj', self.em['F_munu'], self.em['F_munu']) * (1/(4*np.pi*self.eps_0))
        T_EM += CONFIG["j4_coupling_factor"] * J4[:, np.newaxis, np.newaxis] * np.eye(4)
        T += T_EM
        T_strong = np.einsum('...aij,...akj', self.strong['F_munu'], self.strong['F_munu']) * self.g_strong
        T += T_strong
        T_weak = np.einsum('...aij,...akj', self.weak['W_munu'], self.weak['W_munu']) * self.g_weak
        T += T_weak
        self.stress_energy = T
        return T

    def evolve_system(self, steps=CONFIG["max_iterations"]):
        """Evolves the entire system over a number of steps."""
        swarm = [{"state": TARGET_PHYSICAL_STATE + i, "temporal_pos": time.perf_counter_ns() / 1e9} 
                 for i in range(CONFIG["swarm_size"])]
        for step in range(steps):
            current_time = time.perf_counter_ns() / 1e9
            self.iteration = step
            self.quantum_walk(step, current_time)
            for particle in swarm:
                particle["fitness"], _, _ = self.compute_fitness(particle["state"], particle["temporal_pos"])
                particle["state"] = (particle["state"] + repeating_curve(step)) % 2**32
                particle["temporal_pos"] = current_time
            self.evolve_quantum_field_rk4()
            for i in range(self.resolution):
                for μ in range(4):
                    for ν in range(4):
                        self.em['F_munu'][i, μ, ν] = (self.em['A_mu'][(i+1)%self.resolution, ν] - 
                                                      self.em['A_mu'][(i-1)%self.resolution, ν] - 
                                                      self.em['A_mu'][(i+1)%self.resolution, μ] + 
                                                      self.em['A_mu'][(i-1)%self.resolution, μ]) / (2*self.dx)
                self.em['A_mu'][i] += CONFIG["j4_coupling_factor"] * self.em['J4'][i] * np.random.normal(0, 1, 4)
            f_abc = np.zeros((8, 8, 8))
            f_abc[0, 1, 2] = 1; f_abc[0, 2, 1] = -1
            for i in range(self.resolution):
                for a in range(8):
                    for μ in range(4):
                        for ν in range(4):
                            dA_mu = (self.strong['A_mu'][(i+1)%self.resolution, a, ν] - 
                                     self.strong['A_mu'][(i-1)%self.resolution, a, ν]) / (2*self.dx)
                            dA_nu = (self.strong['A_mu'][(i+1)%self.resolution, a, μ] - 
                                     self.strong['A_mu'][(i-1)%self.resolution, a, μ]) / (2*self.dx)
                            nonlinear = self.g_strong * np.sum(f_abc[a] * self.strong['A_mu'][i, :, μ] * self.strong['A_mu'][i, :, ν])
                            self.strong['F_munu'][i, a, μ, ν] = dA_mu - dA_nu + nonlinear
            ε_abc = np.zeros((3, 3, 3))
            ε_abc[0, 1, 2] = 1; ε_abc[1, 2, 0] = 1; ε_abc[2, 0, 1] = 1
            ε_abc[0, 2, 1] = -1; ε_abc[1, 0, 2] = -1; ε_abc[2, 1, 0] = -1
            for i in range(self.resolution):
                for a in range(3):
                    for μ in range(4):
                        for ν in range(4):
                            dW_mu = (self.weak['W_mu'][(i+1)%self.resolution, a, ν] - 
                                     self.weak['W_mu'][(i-1)%self.resolution, a, ν]) / (2*self.dx)
                            dW_nu = (self.weak['W_mu'][(i+1)%self.resolution, a, μ] - 
                                     self.weak['W_mu'][(i-1)%self.resolution, a, μ]) / (2*self.dx)
                            nonlinear = self.g_weak * np.sum(ε_abc[a] * self.weak['W_mu'][i, :, μ] * self.weak['W_mu'][i, :, ν])
                            self.weak['W_munu'][i, a, μ, ν] = dW_mu - dW_nu + nonlinear
            t = self.fabric[:, 3] + self.time
            self.gw['plus'] = 1e-6 * np.sin(2 * np.pi * 7.83 * t)
            self.gw['cross'] = 1e-6 * np.cos(2 * np.pi * 7.83 * t)
            self.move_charged_particles(self.dt)
            self.evolve_phi_N()
            self.compute_maxwell_equations()
            self.compute_delta_g_tphi()
            self.metric = self.compute_metric()
            self.christoffel = self.compute_christoffel_symbols_analytical()
            self.total_stress_energy()
            flux_signal = self.generate_flux_signal()
            self.activate_flux_capacitor(flux_signal)
            time.sleep(0.001)

    def visualize_unified_fields(self):
        """Visualizes the unified fields and particle paths."""
        fig = plt.figure(figsize=(20, 16))
        
        ax1 = fig.add_subplot(221, projection='3d')
        x, y, z = self.fabric[:, :3].T
        ricci_proxy = np.diag(self.metric[:, 0, 0])
        sc1 = ax1.scatter(x, y, z, c=ricci_proxy, cmap='viridis')
        ax1.set_title('Spacetime Curvature')
        fig.colorbar(sc1, ax=ax1)

        ax2 = fig.add_subplot(222)
        spinor_norm = np.sum(np.abs(self.quantum['spinor'])**2, axis=1)
        quark_norm = np.sum(np.abs(self.quantum['quark_spinors'])**2, axis=(1, 2, 3))
        ax2.plot(spinor_norm, label='Electron Spinor Norm')
        ax2.plot(quark_norm, label='Quark Spinor Norm')
        ax2.set_title('Quantum Field Densities')
        ax2.legend()

        ax3 = fig.add_subplot(223)
        ax3.plot(self.fabric[:, 3], self.gw['plus'], label='+ Polarization')
        ax3.plot(self.fabric[:, 3], self.gw['cross'], label='× Polarization')
        ax3.set_title('Gravitational Waves')
        ax3.legend()

        ax4 = fig.add_subplot(224, projection='3d')
        for (i, j), _ in self.edges:
            ax4.plot([self.fabric[i, 0], self.fabric[j, 0]], 
                     [self.fabric[i, 1], self.fabric[j, 1]], 
                     [self.fabric[i, 2], self.fabric[j, 2]], 'r-', alpha=0.5)
        for p in self.particles:
            path = np.array(p['path'])
            ax4.plot(path[:, 0], path[:, 1], path[:, 2], c='blue', alpha=0.5)
        ax4.set_title('Spacetime Structure & Particle Paths')
        
        plt.tight_layout()
        plt.show()

    def close(self):
        """Closes the serial connection."""
        if self.arduino and self.arduino.is_open:
            self.arduino.close()
            print("Serial connection closed.")

# Main Execution
if __name__ == "__main__":
    sim = UnifiedSpacetimeSimulator(serial_port='COM3')
    try:
        for v in sim.fabric:
            sim.add_particle(position=v, velocity=0.1 * sim.c * np.random.randn(4), charge=1.6e-19)
        sim.evolve_system(steps=CONFIG["steps"])
        sim.visualize_unified_fields()
    finally:
        sim.close()
