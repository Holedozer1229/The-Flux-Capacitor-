import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.linalg import expm
from scipy.io import wavfile
import serial
import time
import logging

# Configure logging
logging.basicConfig(filename='flux_capacitor_ctc.log', level=logging.INFO,
                    format='%(asctime)s.%(msecs)03d - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger("FluxCapacitorCTC")

# Configuration with units (aligned with manuscript)
CONFIG = {
    "resolution": 20,                     # 20-bit lattice per manuscript, dimensionless
    "lambda_": 2.72,                      # m, characteristic length scale
    "kappa": 0.1,                         # dimensionless coupling constant
    "charge_density": 1e-12,              # C/m^3
    "em_strength": 0.1,                   # V/m, adjusted for ~0.1 Tesla electromagnet
    "j4_coupling_factor": 1e-18,          # m^8 s^4 C^-4
    "time_delay_steps": 3,                # dimensionless, CTC delay
    "ctc_feedback_factor": 5.0,           # dimensionless, per manuscript
    "entanglement_factor": 0.2,           # dimensionless
    "nodes": 16,                          # dimensionless, spin network nodes
    "alpha_em": 1/137,                    # dimensionless, fine-structure constant
    "phi_N_evolution_factor": 1e-6,       # dimensionless
    "steps": 199,                         # dimensionless, per manuscript test
    "swarm_size": 5,                      # dimensionless
    "max_iterations": 200,                # dimensionless
    "charge": 1.0                         # C
}

# Global variables
RS = 2.0  # m, spatial scale
TARGET_PHYSICAL_STATE = int(time.time() * 1000)
START_TIME = time.perf_counter_ns() / 1e9
KNOWN_STATE = int(START_TIME * 1000) % 2**32

def repeating_curve(index):
    return 1 if index % 2 == 0 else 0

class SpinNetwork:
    def __init__(self, nodes=CONFIG["nodes"]):
        self.nodes = nodes
        self.edges = [(i, (i + 1) % nodes) for i in range(nodes)]
        self.state = np.ones(nodes, dtype=complex) / np.sqrt(nodes)

    def evolve(self, H, dt):
        self.state = expm(-1j * H * dt) @ self.state

    def get_adjacency_matrix(self):
        A = np.zeros((self.nodes, self.nodes))
        for i, j in self.edges:
            A[i, j] = A[j, i] = 1
        return A

class CTCTetrahedralField:
    def __init__(self, resolution=CONFIG["resolution"]):
        self.resolution = resolution
        self.coordinates = self._generate_tetrahedral_coordinates()
        self.H = self._build_hamiltonian()

    def _generate_tetrahedral_coordinates(self):
        coords = np.zeros((self.resolution, 4))
        t = np.linspace(0, 2 * np.pi, self.resolution)
        coords[:, 0] = np.cos(t) * np.sin(t)
        coords[:, 1] = np.sin(t) * np.sin(t)
        coords[:, 2] = np.cos(t)
        coords[:, 3] = t / (2 * np.pi)
        return coords

    def _build_hamiltonian(self):
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
        return expm(-1j * self.H * τ) @ ψ0

class UnifiedSpacetimeSimulator:
    def __init__(self, resolution=CONFIG["resolution"], lambda_=CONFIG["lambda_"], 
                 kappa=CONFIG["kappa"], charge_density=CONFIG["charge_density"],
                 serial_port=None, baud_rate=115200):
        # Physical constants
        self.c = 3e8                          # m/s
        self.G = 6.67430e-11                  # m^3 kg^-1 s^-2
        self.eps_0 = 8.854187817e-12          # F/m
        self.hbar = 1.0545718e-34             # J s
        self.mass_e = 9.11e-31                # kg
        self.mass_q = 2.2e-30                 # kg
        self.g_strong = 1.0                   # dimensionless
        self.g_weak = 0.653                   # dimensionless

        # Simulation parameters
        self.resolution = resolution
        self.lambda_ = lambda_                # m
        self.kappa = kappa                    # dimensionless
        self.charge_density = charge_density  # C/m^3
        self.dx = 4.0 / (resolution - 1)      # m
        self.dt = self.dx / (2 * self.c)      # s (CFL condition)
        self.time = 0.0                       # s

        # Audio and modulation parameters
        self.schumann_freqs = [7.83, 14.3, 20.8, 27.3, 33.8]  # Hz, per manuscript
        self.flux_freq = 0.00083              # Hz, track-switch frequency
        self.schumann_amplitudes = [1.0, 0.5, 0.33, 0.25, 0.2]  # dimensionless
        self.pythagorean_ratios = [1.0, 2.0, 3/2, 4/3]  # dimensionless
        self.all_flux_signals = []

        # Hardware integration
        self.arduino = None
        if serial_port:
            try:
                self.arduino = serial.Serial(serial_port, baud_rate, timeout=1)
                time.sleep(2)  # Allow Arduino to initialize
                print(f"Connected to Arduino on {serial_port}")
            except serial.SerialException as e:
                print(f"Error connecting to Arduino: {e}")

        # Components
        self.spin_network = SpinNetwork()
        self.tetrahedral_field = CTCTetrahedralField()
        self.quantum_state = np.ones(resolution, dtype=complex) / np.sqrt(resolution)
        self.bit_states = np.array([repeating_curve(i) for i in range(resolution)], dtype=int)
        self.temporal_entanglement = np.zeros(resolution)
        self.history = []
        self.bit_flip_log = []
        self.j4_effect_log = []
        self.phi_N_history = []
        self.j4_history = []
        self.flux_amplitude_history = []
        self.steps = []

        # Fields and particles
        self.fabric, self.edges = self.generate_spacetime_fabric()
        self.quantum = self.init_quantum_fields()
        self.em = self.init_em_fields()
        self.strong = self.init_strong_fields()
        self.weak = self.init_weak_fields()
        self.gw = self.init_gravitational_waves()
        self.stress_energy = np.zeros((resolution, 4, 4), dtype=np.float32)
        self.particles = []
        self.iteration = 0
        self.ctc_influence = 0.0
        self.delta_g_tphi = np.zeros(resolution)
        self.metric = self.compute_metric()
        self.christoffel = self.compute_christoffel_symbols()

        logger.info(f"Initialized at Time {int(self.time * 1e9)} ns")

    def generate_spacetime_fabric(self):
        scale = RS  # m, per manuscript
        vertices = []
        for i in range(self.resolution):
            theta = 2 * np.pi * i / self.resolution
            phi = np.pi * i / self.resolution
            x = scale * np.sin(phi) * np.cos(theta)
            y = scale * np.sin(phi) * np.sin(theta)
            z = scale * np.cos(phi)
            t = i * self.dt
            vertices.append(np.array([x, y, z, t], dtype=np.float32))
        edges = [(i, (i + 1) % self.resolution) for i in range(self.resolution)]
        spins = np.random.choice([0.5, 1.0], len(edges))
        return np.array(vertices), list(zip(edges, spins))

    def init_quantum_fields(self):
        r = np.linalg.norm(self.fabric[:, :3], axis=1)  # m
        phi_N = (self.lambda_ / self.kappa) * (1 - np.exp(-r**2 / self.lambda_**2))  # dimensionless
        return {
            'spinor': np.random.normal(0, 1e-3, (self.resolution, 4)).astype(np.complex64),
            'quark_spinors': np.random.normal(0, 1e-3, (self.resolution, 2, 3, 4)).astype(np.complex64),
            'phi_N': phi_N.astype(np.float32)
        }

    def init_em_fields(self):
        em = {
            'A_mu': np.zeros((self.resolution, 4), dtype=np.float32),  # V, V s/m
            'F_munu': np.zeros((self.resolution, 4, 4), dtype=np.float32),  # V/m, T
            'J': np.zeros((self.resolution, 4), dtype=np.float32),  # C/m^2 s
            'J4': np.zeros(self.resolution, dtype=np.float32),  # (C/m^2 s)^4
            'charge': self.charge_density * (self.dx**3)  # C
        }
        em['J'][:, 0] = self.charge_density * self.c
        em['J4'] = np.power(np.linalg.norm(em['J'], axis=1), 4)
        return em

    def init_strong_fields(self):
        return {
            'A_mu': np.zeros((self.resolution, 8, 4), dtype=np.float32),
            'F_munu': np.zeros((self.resolution, 8, 4, 4), dtype=np.float32)
        }

    def init_weak_fields(self):
        return {
            'W_mu': np.zeros((self.resolution, 3, 4), dtype=np.float32),
            'W_munu': np.zeros((self.resolution, 3, 4, 4), dtype=np.float32),
            'higgs': np.ones(self.resolution, dtype=np.float32) * 246e9 / self.c
        }

    def init_gravitational_waves(self):
        t = self.fabric[:, 3]
        return {
            'plus': 1e-6 * np.sin(2 * np.pi * self.schumann_freqs[0] * t),
            'cross': 1e-6 * np.cos(2 * np.pi * self.schumann_freqs[0] * t)
        }

    def compute_metric(self):
        metric = np.tile(np.eye(4, dtype=np.float32)[np.newaxis, :, :], (self.resolution, 1, 1))
        r = np.linalg.norm(self.fabric[:, :3], axis=1)
        r = np.where(r < 1e-6, 1e-6, r)
        phi_N = self.quantum['phi_N'][:, np.newaxis, np.newaxis]
        J4 = self.em['J4'][:, np.newaxis, np.newaxis]
        mass = 1.989e30  # kg
        mass_term = 2 * self.G * mass / (self.c**2 * r)
        charge_term = (self.G * self.em['charge']**2) / (4 * np.pi * self.eps_0 * self.c**4 * r**2)
        j4_term = CONFIG["j4_coupling_factor"] * J4 * (self.c**4 / self.G)
        for i in range(self.resolution):
            metric[i, 0, 0] = -(1 - mass_term[i] + charge_term[i] + j4_term[i]) * (1 + self.kappa * phi_N[i])
            metric[i, 1, 1] = (1 + self.kappa * phi_N[i]) / (1 - mass_term[i] + charge_term[i] + j4_term[i])
            metric[i, 2, 2] = r[i]**2 * (1 + self.kappa * phi_N[i] + j4_term[i])
            metric[i, 3, 3] = metric[i, 2, 2] * np.sin(np.arctan2(self.fabric[i, 1], self.fabric[i, 0]))**2
            metric[i, 0, 3] = metric[i, 3, 0] = CONFIG["alpha_em"] * self.delta_g_tphi[i]
        return metric

    def compute_inverse_metric(self):
        g_inv = np.zeros_like(self.metric)
        for i in range(self.resolution):
            try:
                g_inv[i] = np.linalg.inv(self.metric[i])
            except np.linalg.LinAlgError:
                g_inv[i] = np.eye(4)
        return g_inv

    def compute_christoffel_symbols(self):
        christoffel = np.zeros((self.resolution, 4, 4, 4), dtype=np.float32)
        g_inv = self.compute_inverse_metric()
        for i in range(self.resolution):
            if np.linalg.norm(self.fabric[i, :3]) < 1e-6:
                continue
            for lam in range(4):
                for mu in range(4):
                    for nu in range(4):
                        sum_sigma = 0.0
                        for sigma in range(4):
                            i_plus = (i + 1) % self.resolution
                            i_minus = (i - 1) % self.resolution
                            dg_dmu = (self.metric[i_plus, sigma, nu] - self.metric[i_minus, sigma, nu]) / (2 * self.dx)
                            dg_dnu = (self.metric[i_plus, mu, sigma] - self.metric[i_minus, mu, sigma]) / (2 * self.dx)
                            dg_dsigma = (self.metric[i_plus, mu, nu] - self.metric[i_minus, mu, nu]) / (2 * self.dx)
                            sum_sigma += g_inv[i, lam, sigma] * (dg_dmu + dg_dnu - dg_dsigma)
                        christoffel[i, lam, mu, nu] = 0.5 * sum_sigma
        return christoffel

    def compute_ricci(self):
        R = np.zeros((self.resolution, 4, 4), dtype=np.float32)
        eigenvalues = np.zeros((self.resolution, 4), dtype=np.float32)
        Gamma = self.christoffel
        h = 1e-4
        for i in range(self.resolution):
            if np.linalg.norm(self.fabric[i, :3]) < 1e-6:
                continue
            for mu in range(4):
                for nu in range(4):
                    term1 = term2 = term3 = term4 = 0
                    for lam in range(4):
                        dGamma_lam = (Gamma[(i+1)%self.resolution, lam, mu, nu] - 
                                      Gamma[(i-1)%self.resolution, lam, mu, nu]) / (2*h + 1e-6)
                        term1 += dGamma_lam
                        dGamma_nu = (Gamma[(i+1)%self.resolution, lam, mu, lam] - 
                                     Gamma[(i-1)%self.resolution, lam, mu, lam]) / (2*h + 1e-6)
                        term2 -= dGamma_nu
                    for lam in range(4):
                        for sig in range(4):
                            term3 += Gamma[i, lam, mu, nu] * Gamma[i, sig, lam, sig]
                            term4 -= Gamma[i, lam, mu, sig] * Gamma[i, sig, nu, lam]
                    R[i, mu, nu] = term1 + term2 + term3 + term4
            try:
                eigvals = np.linalg.eigvals(R[i])
                eigenvalues[i] = np.sort(np.real(eigvals))
            except np.linalg.LinAlgError:
                eigenvalues[i] = np.zeros(4)
        return R, eigenvalues

    def evolve_phi_N(self):
        F_munu = self.em['F_munu']
        J4 = self.em['J4']
        F_squared = np.einsum('ijk,ijk->i', F_munu, F_munu)
        delta_phi_N = CONFIG["phi_N_evolution_factor"] * (F_squared + CONFIG["j4_coupling_factor"] * J4) * self.dt
        self.quantum['phi_N'] += delta_phi_N  # No clipping per manuscript
        return self.quantum['phi_N']

    def compute_lagrangian(self):
        F_munu = self.em['F_munu']
        J4 = self.em['J4']
        F_squared = np.einsum('ijk,ijk->i', F_munu, F_munu)
        self.lagrangian = -0.25 * F_squared / (4 * np.pi * self.eps_0) + (CONFIG["j4_coupling_factor"] / 4) * J4
        return self.lagrangian

    def compute_maxwell_equations(self):
        F_munu = self.em['F_munu'].copy()
        J_nu = self.em['J']
        J4 = self.em['J4']
        g_inv = self.compute_inverse_metric()
        F_upper = np.einsum('ijk,ikl,imn->ijn', g_inv, g_inv, F_munu)
        div_F = np.zeros((self.resolution, 4), dtype=np.float32)
        for i in range(self.resolution):
            if np.linalg.norm(self.fabric[i, :3]) < 1e-6:
                continue
            for nu in range(4):
                for mu in range(4):
                    dF = (F_upper[(i + 1) % self.resolution, mu, nu] - F_upper[(i - 1) % self.resolution, mu, nu]) / (2 * self.dx)
                    div_F[i, nu] += dF + np.sum(self.christoffel[i, mu, :, :] * F_upper[i, :, nu])
        for i in range(self.resolution):
            for nu in range(4):
                source = 4 * np.pi * J_nu[i, nu] / self.eps_0 + CONFIG["j4_coupling_factor"] * J4[i]
                discrepancy = (source - div_F[i, nu]) * self.dt
                for mu in range(4):
                    F_munu[i, mu, nu] += discrepancy
                    F_munu[i, nu, mu] = -F_munu[i, mu, nu]
        self.em['F_munu'] = F_munu
        return div_F

    def compute_delta_g_tphi(self):
        F_munu = self.em['F_munu']
        J4 = self.em['J4']
        g_inv = self.compute_inverse_metric()
        F_squared = np.einsum('ijk,ikl,imn->i', F_munu, g_inv, F_munu) + CONFIG["j4_coupling_factor"] * J4
        self.delta_g_tphi += CONFIG["alpha_em"] * F_squared * self.dt
        return self.delta_g_tphi

    def compute_vector_potential(self, iteration):
        A = np.zeros((self.resolution, 4))
        r = np.linalg.norm(self.fabric[:, :3], axis=1)
        theta = np.arctan2(self.fabric[:, 1], self.fabric[:, 0])
        J4 = self.em['J4']
        load_factor = (time.perf_counter_ns() / 1e9 - START_TIME) / 5
        for i in range(self.resolution):
            j4_effect = CONFIG["j4_coupling_factor"] * J4[i]
            A[i, 0] = -CONFIG["charge"] / (4 * np.pi * self.eps_0 * (r[i] + 1e-8)) * (1 + np.sin(iteration * 0.2) * load_factor + j4_effect)
            A[i, 3] = (CONFIG["em_strength"] / self.c) * r[i] * np.sin(theta[i]) * (1 + load_factor + j4_effect)
        return A

    def quantum_walk(self, iteration, current_time):
        A_mu = self.compute_vector_potential(iteration)
        self.em['A_mu'] = A_mu
        prob = np.abs(self.quantum_state)**2
        adj_matrix = self.spin_network.get_adjacency_matrix()
        self.spin_network.evolve(adj_matrix, self.dt)
        J_eff = CONFIG["j4_coupling_factor"] * self.em['J4']
        for idx in range(self.resolution):
            expected_state = repeating_curve(idx + iteration)
            self.bit_states[idx] = expected_state
            window = prob[max(0, idx - CONFIG["time_delay_steps"]):idx + 1]
            self.temporal_entanglement[idx] = CONFIG["entanglement_factor"] * np.mean(window) if window.size > 0 else 0
            em_pert = CONFIG["em_strength"] * np.linalg.norm(self.em['A_mu'][idx])
            j4_pert = J_eff[idx]
            if np.random.random() < (em_pert + j4_pert) * self.temporal_entanglement[idx]:
                self.bit_states[idx] = 1 - self.bit_states[idx]
                self.bit_flip_log.append((current_time, idx, self.bit_states[idx]))
        self.quantum_state = self.tetrahedral_field.propagate(self.quantum_state, 2 * np.pi / self.resolution)
        self.history.append((int(current_time * 1e9), self.bit_states.copy()))
        em_effect = np.mean(np.abs(A_mu[:, 0]))
        logger.info(f"Iteration {iteration}, Time {int(current_time * 1e9)}: Bit States = {self.bit_states.tolist()}, "
                    f"Entanglement = {self.temporal_entanglement[0]:.4f}, EM Effect = {em_effect:.6f}")

    def compute_fitness(self, state, temporal_pos):
        current_time = time.perf_counter_ns() / 1e9
        delta_time = current_time - temporal_pos
        base_fitness = abs(state - KNOWN_STATE)
        ctc_influence = 0
        if self.iteration >= CONFIG["time_delay_steps"] and len(self.history) >= CONFIG["time_delay_steps"]:
            past_states = [h[1][0] for h in self.history[-CONFIG["time_delay_steps"]:]]
            ctc_influence = np.mean(past_states) * CONFIG["ctc_feedback_factor"]
            self.ctc_influence = 1.6667 if self.iteration % 2 == 0 else 3.3333
        j4_effect = CONFIG["j4_coupling_factor"] * np.mean(self.em['J4'])
        fitness = base_fitness + ctc_influence + j4_effect
        return fitness, delta_time, ctc_influence

    def generate_fourier_borel_signal(self, t, num_terms=10):
        f_signal = np.zeros_like(t, dtype=np.float32)
        for n in range(1, num_terms + 1):
            k = 2 * n - 1
            x = 2 * np.pi * self.flux_freq * t
            f_signal += (1 / k) * np.sin(k * x)
        f_signal *= (4 / np.pi)
        return f_signal

    def generate_flux_signal(self, duration=1.0, sample_rate=22050, num_fourier_terms=10):
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        flux_signal = np.zeros_like(t, dtype=np.float32)
        base_signal = np.sin(2 * np.pi * self.flux_freq * t) * 0.5
        total_amplitude_mod = 0.0
        total_freq_mod = 0.0
        for i in range(self.resolution):
            A_mu_norm = np.linalg.norm(self.em['A_mu'][i]) * 10
            gw_plus = self.gw['plus'][i] * 1000
            gw_cross = self.gw['cross'][i] * 1000
            total_amplitude_mod += (A_mu_norm + gw_plus + gw_cross) * 0.01
            j4_effect = self.em['J4'][i] * CONFIG["j4_coupling_factor"] * 1e-3
            phi_N_effect = self.quantum['phi_N'][i] * 1e-2
            total_freq_mod += (j4_effect + phi_N_effect) * 10
        amplitude_mod = 1 + total_amplitude_mod / self.resolution
        freq_mod = self.flux_freq + total_freq_mod / self.resolution
        flux_signal = amplitude_mod * np.sin(2 * np.pi * freq_mod * t) + base_signal
        schumann_mod = sum(A * np.cos(2 * np.pi * f * t) for f, A in zip(self.schumann_freqs, self.schumann_amplitudes))
        flux_signal += schumann_mod * 0.1
        fourier_borel = self.generate_fourier_borel_signal(t, num_fourier_terms)
        flux_signal += fourier_borel * 0.05
        max_abs = np.max(np.abs(flux_signal))
        if max_abs > 0:
            flux_signal /= max_abs
        self.flux_amplitude_history.append(max_abs if max_abs > 0 else 1.0)
        return flux_signal

    def activate_flux_capacitor(self, signal, sample_rate=22050):
        if self.arduino:
            try:
                scaled_signal = ((signal + 1) * 127.5).astype(np.uint8)  # Scale to 0-255 for PWM
                start_time = time.perf_counter_ns() / 1e9
                for idx, value in enumerate(scaled_signal):
                    self.arduino.write(bytes([value]))
                    if self.arduino.in_waiting > 0:
                        feedback = self.arduino.readline().decode().strip()
                        try:
                            hall_value = float(feedback)  # Hall sensor reading (0-5V mapped)
                            self.em['J'][:, 0] += hall_value * 1e-6  # Update current density
                            self.em['J4'] = np.power(np.linalg.norm(self.em['J'], axis=1), 4)
                            logger.info(f"Hall Sensor Feedback: {hall_value}")
                        except ValueError:
                            pass
                    time.sleep(1 / sample_rate)  # Pace to match sample rate
                state = int(np.sum(self.bit_states * (2 ** np.arange(self.resolution))))
                fitness, delta_t, ctc_influence = self.compute_fitness(state, start_time)
                logger.info(f"Iteration {self.iteration}, Time {int(current_time * 1e9)}: "
                            f"Bit States = {self.bit_states.tolist()}, Entanglement = {self.temporal_entanglement[0]:.4f}, "
                            f"State = {state}, Fitness = {fitness:.2f}, DeltaT = {delta_t:.6f}, CTC Influence = {ctc_influence:.4f}")
            except serial.SerialException as e:
                print(f"Error communicating with Arduino: {e}")
        self.all_flux_signals.append(signal)

    def save_combined_wav(self, sample_rate=22050):
        if not self.all_flux_signals:
            print("No flux signals to save.")
            return
        print("Writing combined flux capacitor signal to WAV file...")
        combined_signal = np.concatenate(self.all_flux_signals)
        signal_int16 = np.int16(combined_signal * 32767)
        wavfile.write('flux_signal_combined.wav', sample_rate, signal_int16)
        print("Saved flux_signal_combined.wav")
        self.all_flux_signals = []

    def total_stress_energy(self):
        T = np.zeros((self.resolution, 4, 4), dtype=np.float32)
        T[:, 0, 0] = -self.quantum['phi_N'] / self.c**2
        T_EM = np.einsum('ijk,ikl->ijl', self.em['F_munu'], self.em['F_munu']) / (4 * np.pi * self.eps_0)
        T += T_EM + CONFIG["j4_coupling_factor"] * self.em['J4'][..., np.newaxis, np.newaxis] * np.eye(4)
        T_strong = np.einsum('iajk,iakl->ijl', self.strong['F_munu'], self.strong['F_munu']) * self.g_strong
        T_weak = np.einsum('iajk,iakl->ijl', self.weak['W_munu'], self.weak['W_munu']) * self.g_weak
        self.stress_energy = T + T_strong + T_weak
        return self.stress_energy

    def add_particle(self, position, velocity, charge):
        particle = {
            'position': np.array(position[:3], dtype=np.float32),
            'velocity': np.array(velocity[:3], dtype=np.float32),
            'charge': charge,
            'path': [position[:3].copy()]
        }
        self.particles.append(particle)

    def equations_of_motion(self, y, t):
        x, y, z, vx, vy, vz = y
        i = self.find_closest_fabric_point(x, y, z)
        v = np.array([vx, vy, vz])
        v2 = np.sum(v**2) / self.c**2
        gamma = 1 / np.sqrt(max(1 - v2, 1e-10))
        u = np.array([gamma, gamma * vx / self.c, gamma * vy / self.c, gamma * vz / self.c])
        geo_accel = np.zeros(3)
        for dim in range(3):
            Γ = self.christoffel[i, dim + 1, :, :]
            geo_accel[dim] = -np.einsum('ij,i,j', Γ, u, u) * self.c**2
        q = self.particles[0]['charge']
        F_upper = np.einsum('ij,jk->ik', self.compute_inverse_metric()[i], self.em['F_munu'][i])
        em_force = q * np.einsum('ij,j->i', F_upper[1:4, :], u)
        strong_force = np.zeros(3)
        for a in range(8):
            strong_force += self.g_strong * np.einsum('ij,j', self.strong['F_munu'][i, a, 1:4, :], u)
        weak_force = np.zeros(3)
        for a in range(3):
            weak_force += self.g_weak * np.einsum('ij,j', self.weak['W_munu'][i, a, 1:4, :], u)
        j4_force = CONFIG["j4_coupling_factor"] * self.em['J4'][i] * u[1:4]
        accel = (geo_accel + em_force + strong_force + weak_force + j4_force) / self.mass_e
        return [vx, vy, vz, accel[0], accel[1], accel[2]]

    def move_charged_particles(self, dt):
        for p in self.particles:
            state = np.concatenate([p['position'], p['velocity']])
            t_range = np.linspace(0, dt, 10)
            trajectory = odeint(self.equations_of_motion, state, t_range)
            p['position'] = trajectory[-1, :3]
            p['velocity'] = trajectory[-1, 3:]
            p['path'].append(p['position'].copy())

    def find_closest_fabric_point(self, x, y, z):
        r_point = np.sqrt(x**2 + y**2 + z**2)
        return np.argmin(np.abs(np.linalg.norm(self.fabric[:, :3], axis=1) - r_point))

    def evolve_quantum_field_rk4(self):
        dt = self.dt
        for i in range(self.resolution):
            t = self.time
            psi = self.quantum['spinor'][i].copy()
            k1 = -1j * self.dirac_hamiltonian(psi, i, t) / self.hbar
            k2 = -1j * self.dirac_hamiltonian(psi + 0.5 * dt * k1, i, t + 0.5 * dt) / self.hbar
            k3 = -1j * self.dirac_hamiltonian(psi + 0.5 * dt * k2, i, t + 0.5 * dt) / self.hbar
            k4 = -1j * self.dirac_hamiltonian(psi + dt * k3, i, t + dt) / self.hbar
            self.quantum['spinor'][i] += (k1 + 2 * k2 + 2 * k3 + k4) * (dt / 6)
            for flavor in range(2):
                for color in range(3):
                    psi_q = self.quantum['quark_spinors'][i, flavor, color].copy()
                    k1_q = -1j * self.dirac_hamiltonian(psi_q, i, t, quark=True, flavor=flavor, color=color) / self.hbar
                    k2_q = -1j * self.dirac_hamiltonian(psi_q + 0.5 * dt * k1_q, i, t + 0.5 * dt, quark=True, flavor=flavor, color=color) / self.hbar
                    k3_q = -1j * self.dirac_hamiltonian(psi_q + 0.5 * dt * k2_q, i, t + 0.5 * dt, quark=True, flavor=flavor, color=color) / self.hbar
                    k4_q = -1j * self.dirac_hamiltonian(psi_q + dt * k3_q, i, t + dt, quark=True, flavor=flavor, color=color) / self.hbar
                    self.quantum['quark_spinors'][i, flavor, color] += (k1_q + 2 * k2_q + 2 * k3_q + k4_q) * (dt / 6)
        self.time += dt

    def dirac_hamiltonian(self, psi, i, t, quark=False, flavor=None, color=None):
        gamma_mu = self.dirac_gamma_matrices(self.metric[i])
        mass = self.mass_q if quark else self.mass_e
        H_psi = np.zeros(4, dtype=np.complex64)
        for mu in range(1, 4):
            D_mu_psi = self.covariant_derivative(psi, i, mu)
            H_psi -= 1j * self.c * gamma_mu[0] @ gamma_mu[mu] @ D_mu_psi
        H_psi += (mass * self.c**2 / self.hbar) * gamma_mu[0] @ psi
        H_psi -= 1j * self.em['charge'] * np.sum([self.em['A_mu'][i, mu] * gamma_mu[mu] @ psi for mu in range(4)], axis=0)
        if quark and flavor is not None and color is not None:
            T_a = self.gell_mann_matrices()
            strong_term = np.zeros(4, dtype=np.complex64)
            for a in range(8):
                for mu in range(4):
                    strong_term += self.strong['A_mu'][i, a, mu] * (T_a[a] @ self.quantum['quark_spinors'][i, flavor, color])
            H_psi += -1j * self.g_strong * strong_term
        H_psi += CONFIG["j4_coupling_factor"] * self.em['J4'][i] * psi
        H_psi += self.schumann_potential(t) * psi
        return H_psi

    def covariant_derivative(self, psi, i, mu):
        if mu != 0:
            i_plus = (i + 1) % self.resolution
            i_minus = (i - 1) % self.resolution
            partial_psi = (self.quantum['spinor'][i_plus] - self.quantum['spinor'][i_minus]) / (2 * self.dx)
            harmonic_scale = self.pythagorean_ratios[mu % len(self.pythagorean_ratios)]
            partial_psi *= harmonic_scale
        else:
            partial_psi = np.zeros_like(psi)
        connection_term = np.einsum('ij,j', self.christoffel[i, mu, :, :], psi)
        return partial_psi + connection_term

    def dirac_gamma_matrices(self, g_mu_nu):
        gamma_flat = [
            np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]], dtype=np.complex64),
            np.array([[0, 0, 0, 1], [0, 0, 1, 0], [0, -1, 0, 0], [-1, 0, 0, 0]], dtype=np.complex64),
            np.array([[0, 0, 0, -1j], [0, 0, 1j, 0], [0, 1j, 0, 0], [-1j, 0, 0, 0]], dtype=np.complex64),
            np.array([[0, 0, 1, 0], [0, 0, 0, -1], [-1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.complex64)
        ]
        e_a_mu = np.diag([np.sqrt(abs(g_mu_nu[i, i])) for i in range(4)])
        e_mu_a = np.linalg.inv(e_a_mu)
        gamma_mu = [sum(e_mu_a[mu, a] * gamma_flat[a] for a in range(4)) for mu in range(4)]
        return gamma_mu

    def gell_mann_matrices(self):
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
        V_0 = 1e-6
        V_t = 0
        for fn, An in zip(self.schumann_freqs, self.schumann_amplitudes):
            V_t += V_0 * An * np.cos(2 * np.pi * fn * t)
        return V_t

    def evolve_system(self, steps=CONFIG["steps"]):
        swarm = [{"state": TARGET_PHYSICAL_STATE + i, "temporal_pos": time.perf_counter_ns() / 1e9} 
                 for i in range(CONFIG["swarm_size"])]
        f_abc = np.zeros((8, 8, 8))
        f_abc[0, 1, 2] = 1; f_abc[0, 2, 1] = -1
        ε_abc = np.zeros((3, 3, 3))
        ε_abc[0, 1, 2] = 1; ε_abc[1, 2, 0] = 1; ε_abc[2, 0, 1] = 1
        ε_abc[0, 2, 1] = -1; ε_abc[1, 0, 2] = -1; ε_abc[2, 1, 0] = -1

        for step in range(steps):
            current_time = time.perf_counter_ns() / 1e9
            self.iteration = step
            self.quantum_walk(step, current_time)
            for particle in swarm:
                particle["fitness"], _, _ = self.compute_fitness(particle["state"], particle["temporal_pos"])
                particle["state"] = (particle["state"] + repeating_curve(step)) % 2**32
                particle["temporal_pos"] = current_time
            self.evolve_quantum_field_rk4()
            self.evolve_phi_N()
            self.compute_maxwell_equations()
            self.compute_delta_g_tphi()
            self.compute_lagrangian()
            self.total_stress_energy()
            for i in range(self.resolution):
                for mu in range(4):
                    nonlinear_strong = self.g_strong * np.sum(f_abc[:, :, mu] * self.strong['A_mu'][i, :, mu, np.newaxis] * self.strong['A_mu'][i, :, mu, np.newaxis])
                    nonlinear_weak = self.g_weak * np.sum(ε_abc[:, :, mu] * self.weak['W_mu'][i, :, mu, np.newaxis] * self.weak['W_mu'][i, :, mu, np.newaxis])
                    self.strong['A_mu'][i, :, mu] += nonlinear_strong * self.dt
                    self.weak['W_mu'][i, :, mu] += nonlinear_weak * self.dt
                for mu in range(4):
                    for nu in range(4):
                        self.em['F_munu'][i, mu, nu] = (
                            (self.em['A_mu'][(i+1)%self.resolution, nu] - self.em['A_mu'][(i-1)%self.resolution, nu]) -
                            (self.em['A_mu'][(i+1)%self.resolution, mu] - self.em['A_mu'][(i-1)%self.resolution, mu])
                        ) / (2 * self.dx) + CONFIG["j4_coupling_factor"] * self.em['J4'][i]
                        for a in range(8):
                            dA_mu_s = (self.strong['A_mu'][(i+1)%self.resolution, a, nu] - self.strong['A_mu'][(i-1)%self.resolution, a, nu]) / (2*self.dx)
                            dA_nu_s = (self.strong['A_mu'][(i+1)%self.resolution, a, mu] - self.strong['A_mu'][(i-1)%self.resolution, a, mu]) / (2*self.dx)
                            nonlinear_s = self.g_strong * np.sum(f_abc[a] * self.strong['A_mu'][i, :, mu] * self.strong['A_mu'][i, :, nu])
                            self.strong['F_munu'][i, a, mu, nu] = dA_mu_s - dA_nu_s + nonlinear_s
                        for a in range(3):
                            dW_mu = (self.weak['W_mu'][(i+1)%self.resolution, a, nu] - self.weak['W_mu'][(i-1)%self.resolution, a, nu]) / (2*self.dx)
                            dW_nu = (self.weak['W_mu'][(i+1)%self.resolution, a, mu] - self.weak['W_mu'][(i-1)%self.resolution, a, mu]) / (2*self.dx)
                            nonlinear_w = self.g_weak * np.sum(ε_abc[a] * self.weak['W_mu'][i, :, mu] * self.weak['W_mu'][i, :, nu])
                            self.weak['W_munu'][i, a, mu, nu] = dW_mu - dW_nu + nonlinear_w
            t = self.fabric[:, 3] + self.time
            self.gw['plus'] = 1e-6 * np.sin(2 * np.pi * self.schumann_freqs[0] * t)
            self.gw['cross'] = 1e-6 * np.cos(2 * np.pi * self.schumann_freqs[0] * t)
            self.move_charged_particles(self.dt)
            self.metric = self.compute_metric()
            self.christoffel = self.compute_christoffel_symbols()
            _, ricci_eigenvalues = self.compute_ricci()
            self.phi_N_history.append(self.quantum['phi_N'][0])
            self.j4_history.append(self.em['J4'][0])
            self.steps.append(self.iteration)
            flux_signal = self.generate_flux_signal()
            self.activate_flux_capacitor(flux_signal)
            logger.info(f"Iteration {step}, Time {int(current_time * 1e9)}: Ricci Eigenvalues = {ricci_eigenvalues[0].tolist()}")

    def visualize_unified_fields(self):
        fig = plt.figure(figsize=(20, 16))
        ax1 = fig.add_subplot(221, projection='3d')
        sc1 = ax1.scatter(self.fabric[:, 0], self.fabric[:, 1], self.fabric[:, 2], 
                          c=self.metric[:, 0, 0], cmap='viridis')
        ax1.set_title('Spacetime Curvature (g_00)')
        fig.colorbar(sc1, ax=ax1)
        ax2 = fig.add_subplot(222)
        spinor_norm = np.sum(np.abs(self.quantum['spinor'])**2, axis=1)
        quark_norm = np.sum(np.abs(self.quantum['quark_spinors'])**2, axis=(1, 2, 3))
        ax2.plot(spinor_norm, label='Electron Spinor')
        ax2.plot(quark_norm, label='Quark Spinor')
        ax2.set_title('Quantum Field Norms')
        ax2.legend()
        ax3 = fig.add_subplot(223)
        ax3.plot(self.fabric[:, 3], self.gw['plus'], label='Plus')
        ax3.plot(self.fabric[:, 3], self.gw['cross'], label='Cross')
        ax3.set_title('Gravitational Waves')
        ax3.legend()
        ax4 = fig.add_subplot(224)
        ax4.plot(self.steps, self.phi_N_history, label='phi_N')
        ax4.plot(self.steps, self.j4_history, label='J^4')
        ax4.plot(self.steps, self.flux_amplitude_history, label='Flux Amplitude')
        ax4.set_title('Dynamics Over Steps')
        ax4.legend()
        plt.tight_layout()
        plt.show()

    def close(self):
        if self.arduino and self.arduino.is_open:
            self.arduino.close()
            print("Serial connection closed.")

if __name__ == "__main__":
    sim = UnifiedSpacetimeSimulator(serial_port='COM3')  # Replace with your Arduino port
    try:
        for v in sim.fabric:
            sim.add_particle(position=v, velocity=0.1 * sim.c * np.random.randn(3), charge=1.6e-19)
        sim.evolve_system(steps=CONFIG["steps"])
        sim.save_combined_wav()
        sim.visualize_unified_fields()
    except Exception as e:
        print(f"Error occurred: {e}")
    finally:
        sim.close()
        print("Simulation completed.")
