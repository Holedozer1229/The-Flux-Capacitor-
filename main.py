import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.signal import resample
import sounddevice as sd
import serial
from mpl_toolkits.mplot3d import Axes3D
import warnings
import time
warnings.filterwarnings('ignore')

class UnifiedSpacetimeSimulator:
    def __init__(self, resolution=50, lambda_=1.0, kappa=0.1, charge_density=1e-12, serial_port='COM3', baud_rate=9600):
        """
        Initialize the Flux Capacitor simulator with spacetime and quantum parameters.
        
        Args:
            resolution (int): Number of spacetime lattice points.
            lambda_ (float): Characteristic length scale (m).
            kappa (float): Coupling constant for scalar field.
            charge_density (float): Charge density (C/m^3).
            serial_port (str): Serial port for Arduino communication (e.g., 'COM3' on Windows, '/dev/ttyUSB0' on Linux).
            baud_rate (int): Baud rate for serial communication.
        """
        self.resolution = resolution
        self.lambda_ = lambda_
        self.kappa = kappa
        self.charge_density = charge_density
        self.beta = 0.1  # Immirzi parameter for LQG
        self.g_strong = 1.0  # Strong coupling constant
        self.g_weak = 0.65   # Weak coupling constant
        self.mass_e = 9.11e-31  # Electron mass (kg)
        self.mass_q = 2.3e-30   # Quark mass (e.g., up quark, kg)
        self.c = 3e8            # Speed of light (m/s)
        self.hbar = 1.0545718e-34  # Reduced Planck constant (J·s)
        self.eps_0 = 8.854187817e-12  # Permittivity of free space (F/m)
        self.G = 6.67430e-11    # Gravitational constant (m³/kg/s²)

        self.dx = 4.0 / (self.resolution - 1)  # Spatial step (m)
        self.dt = 1e-3  # Time step (s)
        self.time = 0.0  # Current simulation time (s)

        # Flux Capacitor-specific frequencies
        self.schumann_freqs = [7.83, 14.3, 20.8, 27.3, 33.8]  # Schumann resonances (Hz)
        self.flux_freq = 0.00083  # 8-track track-switch frequency (Hz)
        self.schumann_amplitudes = [1.0, 0.5, 0.33, 0.25, 0.2]  # Relative amplitudes
        self.pythagorean_ratios = [1.0, 2.0, 3/2, 4/3]  # Harmonic ratios

        # Initialize serial communication with Arduino
        try:
            self.arduino = serial.Serial(serial_port, baud_rate, timeout=1)
            time.sleep(2)  # Wait for Arduino to initialize
            print(f"Connected to Arduino on {serial_port}")
        except serial.SerialException as e:
            print(f"Error connecting to Arduino: {e}")
            self.arduino = None

        self.fabric, self.edges = self.generate_spacetime_fabric()
        self.quantum = self.init_quantum_fields()
        self.metric = self.compute_metric()
        self.christoffel = self.compute_christoffel_symbols_analytical()
        self.em = self.init_em_fields()
        self.strong = self.init_strong_fields()
        self.weak = self.init_weak_fields()
        self.gw = self.init_gravitational_waves()
        self.stress_energy = np.zeros((resolution, 4, 4), dtype=np.float32)
        self.particles = []

    def generate_spacetime_fabric(self):
        """Generate a tetrahedral spacetime lattice with edges and spins."""
        scale = 2 / (3 * np.sqrt(2))
        vertices = [
            np.array([3/2*np.sqrt(2) * scale, 3/2*np.sqrt(2) * scale, 3/2*np.sqrt(2) * scale, 0], dtype=np.float32),
            np.array([3/2*np.sqrt(2) * scale, -3/2*np.sqrt(2) * scale, -3/2*np.sqrt(2) * scale, 0], dtype=np.float32),
            np.array([-3/2*np.sqrt(2) * scale, 3/2*np.sqrt(2) * scale, -3/2*np.sqrt(2) * scale, 0], dtype=np.float32),
            np.array([-3/2*np.sqrt(2) * scale, -3/2*np.sqrt(2) * scale, 3/2*np.sqrt(2) * scale, 0], dtype=np.float32),
            np.array([-3/2*np.sqrt(2) * scale, -3/2*np.sqrt(2) * scale, -3/2*np.sqrt(2) * scale, 0], dtype=np.float32),
            np.array([-3/2*np.sqrt(2) * scale, 3/2*np.sqrt(2) * scale, 3/2*np.sqrt(2) * scale, 0], dtype=np.float32),
            np.array([3/2*np.sqrt(2) * scale, -3/2*np.sqrt(2) * scale, 3/2*np.sqrt(2) * scale, 0], dtype=np.float32),
            np.array([3/2*np.sqrt(2) * scale, 3/2*np.sqrt(2) * scale, -3/2*np.sqrt(2) * scale, 0], dtype=np.float32)
        ]
        edges = [(0, 1), (0, 2), (0, 3), (1, 4), (2, 5), (3, 6), (4, 7), (5, 6), (6, 7), (7, 0)]
        spins = np.random.choice([0.5, 1.0], len(edges))
        fabric = np.array(vertices[:self.resolution], dtype=np.float32)
        return fabric, list(zip(edges[:self.resolution], spins))

    def init_quantum_fields(self):
        """Initialize quantum fields (spinors and scalar field)."""
        r = np.linalg.norm(self.fabric[:, :3], axis=1)
        phi_N = (self.lambda_ / self.kappa) * (1 - np.exp(-r**2 / self.lambda_**2))
        return {
            'spinor': np.random.normal(0, 1e-3, (self.resolution, 4)).astype(np.complex64),
            'quark_spinors': np.random.normal(0, 1e-3, (self.resolution, 2, 3, 4)).astype(np.complex64),
            'phi_N': phi_N.astype(np.float32)
        }

    def compute_metric(self):
        """
        Compute the spacetime metric tensor with full complexity.
        Incorporates mass, charge, and scalar field terms.
        """
        metric = np.tile(np.eye(4, dtype=np.float32)[np.newaxis, :, :], (self.resolution, 1, 1))
        r = np.linalg.norm(self.fabric[:, :3], axis=1)
        r = np.where(r < 1e-6, 1e-6, r)
        phi_N = self.quantum['phi_N'][:, np.newaxis, np.newaxis]
        mass_term = 2 * self.G * 1.989e30 / (self.c**2 * r)  # Solar mass as reference
        charge_term = (self.em['charge'] * self.lambda_ / (4 * np.pi * self.eps_0 * r**2))**2
        for i in range(self.resolution):
            metric[i, 0, 0] = -(1 - mass_term[i] + charge_term[i]) * (1 + self.kappa * phi_N[i])
            metric[i, 1, 1] = 1 / (1 - mass_term[i] + charge_term[i])
            metric[i, 2, 2] = r[i]**2 * (1 + self.kappa * phi_N[i])
            metric[i, 3, 3] = metric[i, 2, 2] * np.sin(self.fabric[i, 2])**2
            metric[i, 0, 3] = 0.27 * self.lambda_ * np.exp(-r[i] / self.lambda_)
            metric[i, 3, 0] = metric[i, 0, 3]
        return metric

    def compute_christoffel_symbols_analytical(self, h=1e-4):
        """Compute Christoffel symbols analytically for geodesic equations."""
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

    def init_em_fields(self):
        """Initialize electromagnetic fields."""
        return {
            'A_mu': np.zeros((self.resolution, 4), dtype=np.float32),
            'F_munu': np.zeros((self.resolution, 4, 4), dtype=np.float32),
            'charge': 1.602e-19  # Electron charge (C)
        }

    def init_strong_fields(self):
        """Initialize strong interaction fields (gluons)."""
        return {
            'A_mu': np.zeros((self.resolution, 8, 4), dtype=np.float32),
            'F_munu': np.zeros((self.resolution, 8, 4, 4), dtype=np.float32)
        }

    def init_weak_fields(self):
        """Initialize weak interaction fields (W and Z bosons, Higgs)."""
        return {
            'W_mu': np.zeros((self.resolution, 3, 4), dtype=np.float32),
            'W_munu': np.zeros((self.resolution, 3, 4, 4), dtype=np.float32),
            'higgs': np.ones(self.resolution, dtype=np.float32) * 246e9 / self.c  # Higgs VEV (GeV/c)
        }

    def init_gravitational_waves(self):
        """Initialize gravitational wave polarizations with Schumann frequency."""
        t = self.fabric[:, 3]
        f_schumann = 7.83  # Hz
        return {
            'plus': 1e-6 * np.sin(2 * np.pi * f_schumann * t),
            'cross': 1e-6 * np.cos(2 * np.pi * f_schumann * t)
        }

    def compute_vierbein(self, g_mu_nu):
        """Compute vierbein for Dirac equation in curved spacetime."""
        e_a_mu = np.zeros((4, 4), dtype=np.float64)
        for mu in range(4):
            e_a_mu[mu, mu] = np.sqrt(np.abs(g_mu_nu[mu, mu]))
        return e_a_mu

    def dirac_gamma_matrices(self, g_mu_nu):
        """Generate Dirac gamma matrices in curved spacetime."""
        gamma_flat = [
            np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]], dtype=np.complex64),
            np.array([[0, 0, 0, 1], [0, 0, 1, 0], [0, -1, 0, 0], [-1, 0, 0, 0]], dtype=np.complex64),
            np.array([[0, 0, 0, -1j], [0, 0, 1j, 0], [0, 1j, 0, 0], [-1j, 0, 0, 0]], dtype=np.complex64),
            np.array([[0, 0, 1, 0], [0, 0, 0, -1], [-1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.complex64)
        ]
        e_a_mu = self.compute_vierbein(g_mu_nu)
        e_mu_a = np.linalg.inv(e_a_mu)
        gamma_mu = [sum(e_mu_a[mu, a] * gamma_flat[a] for a in range(4)) for mu in range(4)]
        return gamma_mu

    def gell_mann_matrices(self):
        """Generate Gell-Mann matrices for SU(3) strong interactions."""
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

    def covariant_derivative(self, psi, i, mu):
        """Compute covariant derivative for spinors in curved spacetime."""
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

    def schumann_potential(self, t):
        """Compute Schumann resonance potential at time t."""
        V_0 = 1e-6  # Base amplitude (arbitrary units)
        V_t = 0
        for fn, An in zip(self.schumann_freqs, self.schumann_amplitudes):
            V_t += V_0 * An * np.cos(2 * np.pi * fn * t)
        return V_t

    def dirac_hamiltonian(self, psi, i, t, quark=False, flavor=None, color=None):
        """Compute Dirac Hamiltonian for electron or quark fields."""
        gamma_mu = self.dirac_gamma_matrices(self.metric[i])
        H_psi = np.zeros(4, dtype=np.complex64)
        mass = self.mass_q if quark else self.mass_e
        for mu in range(1, 4):
            D_mu_psi = self.covariant_derivative(psi, i, mu)
            H_psi -= 1j * self.c * gamma_mu[0] @ gamma_mu[mu] @ D_mu_psi
        H_psi += (mass * self.c**2 / self.hbar) * gamma_mu[0] @ psi
        H_psi -= 1j * self.em['charge'] * sum(self.em['A_mu'][i, mu] * gamma_mu[mu] @ psi for mu in range(4))
        
        V_schumann = self.schumann_potential(t)
        H_psi += V_schumann * psi
        
        if quark and flavor is not None and color is not None:
            T_a = self.gell_mann_matrices()
            strong_term = -1j * self.g_strong * sum(self.strong['A_mu'][i, a, mu] * T_a[a] @ psi 
                                                    for a in range(8) for mu in range(4))
            H_psi += strong_term
        return H_psi

    def dirac_rhs(self, psi, t, i, quark=False, flavor=None, color=None):
        """Right-hand side of Dirac equation for time evolution."""
        return -1j * self.dirac_hamiltonian(psi, i, t, quark, flavor, color) / self.hbar

    def evolve_quantum_field_rk4(self):
        """Evolve quantum fields using 4th-order Runge-Kutta method."""
        dt = self.dt
        for i in range(self.resolution):
            t = self.time
            psi = self.quantum['spinor'][i]
            k1 = self.dirac_rhs(psi, t, i)
            k2 = self.dirac_rhs(psi + 0.5 * dt * k1, t + 0.5 * dt, i)
            k3 = self.dirac_rhs(psi + 0.5 * dt * k2, t + 0.5 * dt, i)
            k4 = self.dirac_rhs(psi + dt * k3, t + dt, i)
            self.quantum['spinor'][i] += (k1 + 2 * k2 + 2 * k3 + k4) * (dt / 6)

            for flavor in range(2):
                for color in range(3):
                    psi_q = self.quantum['quark_spinors'][i, flavor, color]
                    k1_q = self.dirac_rhs(psi_q, t, i, quark=True, flavor=flavor, color=color)
                    k2_q = self.dirac_rhs(psi_q + 0.5 * dt * k1_q, t + 0.5 * dt, i, quark=True, flavor=flavor, color=color)
                    k3_q = self.dirac_rhs(psi_q + 0.5 * dt * k2_q, t + 0.5 * dt, i, quark=True, flavor=flavor, color=color)
                    k4_q = self.dirac_rhs(psi_q + dt * k3_q, t + dt, i, quark=True, flavor=flavor, color=color)
                    self.quantum['quark_spinors'][i, flavor, color] += (k1_q + 2 * k2_q + 2 * k3_q + k4_q) * (dt / 6)
        self.time += dt

    def equations_of_motion(self, y, t):
        """Equations of motion for charged particles in curved spacetime."""
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
        strong_force = self.g_strong * np.sum(self.strong['F_munu'][i, :, 1:4, :], axis=0) @ u[:3]
        weak_force = self.g_weak * np.sum(self.weak['W_munu'][i, :, 1:4, :], axis=0) @ u[:3]
        coordinate_accel = (geo_accel + em_force + strong_force + weak_force) / gamma**2
        coordinate_accel = np.clip(coordinate_accel, -1e10, 1e10)
        return [vx, vy, vz, coordinate_accel[0], coordinate_accel[1], coordinate_accel[2]]

    def find_closest_fabric_point(self, x, y, z):
        """Find the closest lattice point to a given position."""
        r_point = np.sqrt(x**2 + y**2 + z**2)
        i = np.argmin(np.abs(np.linalg.norm(self.fabric[:, :3], axis=1) - r_point))
        return i

    def total_stress_energy(self):
        """Compute the total stress-energy tensor."""
        r = np.linalg.norm(self.fabric[:, :3], axis=1)
        r = np.where(r < 1e-6, 1e-6, r)
        phi_N = self.quantum['phi_N']
        T = np.zeros((self.resolution, 4, 4), dtype=np.float32)
        T[:, 0, 0] = -phi_N / self.c**2
        
        T_EM = np.einsum('...ij,...kj', self.em['F_munu'], self.em['F_munu']) * (1/(4*np.pi*self.eps_0))
        T += T_EM
        
        T_strong = np.einsum('...aij,...akj', self.strong['F_munu'], self.strong['F_munu']) * self.g_strong
        T += T_strong
        
        T_weak = np.einsum('...aij,...akj', self.weak['W_munu'], self.weak['W_munu']) * self.g_weak
        T += T_weak
        
        self.stress_energy = T
        return T

    def generate_fourier_borel_signal(self, t, num_terms=10):
        """
        Generate a Fourier-Borel signal based on the series.
        f(x) = (4/π) Σ_{n=1}^∞ (1/(2n-1)) sin((2n-1)x)
        
        Args:
            t (numpy.ndarray): Time array (s).
            num_terms (int): Number of terms in the Fourier series.
        
        Returns:
            numpy.ndarray: Fourier-Borel signal.
        """
        f_signal = np.zeros_like(t, dtype=np.float32)
        for n in range(1, num_terms + 1):
            k = 2 * n - 1
            x = 2 * np.pi * self.flux_freq * t
            f_signal += (1 / k) * np.sin(k * x)
        f_signal *= (4 / np.pi)
        return f_signal

    def generate_flux_signal(self, duration=10.0, sample_rate=44100, num_fourier_terms=10):
        """
        Generate the Flux Capacitor signal for 8-track modulation, incorporating the Fourier-Borel transform.
        
        Args:
            duration (float): Signal duration (s).
            sample_rate (int): Audio sample rate (Hz).
            num_fourier_terms (int): Number of terms in the Fourier-Borel series.
        
        Returns:
            numpy.ndarray: Normalized flux signal.
        """
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

            harmonic_scale = self.pythagorean_ratios[i % len(self.pythagorean_ratios)]
            scaled_flux = modulated_flux * harmonic_scale

            scaled_flux += fourier_borel

            flux_signal += np.power(np.abs(scaled_flux), 4) * np.sign(scaled_flux)

        flux_signal = np.clip(flux_signal, -1.0, 1.0)
        return flux_signal

    def activate_flux_capacitor(self, signal, sample_rate=44100):
        """
        Activate the Flux Capacitor by outputting the signal to drive the 8-track via Arduino.
        
        Args:
            signal (numpy.ndarray): Flux signal to output.
            sample_rate (int): Audio sample rate (Hz).
        """
        print("Activating Flux Capacitor! Powering 8-track with 1.21 gigawatts of flux...")
        
        # Play the signal through sounddevice for audio monitoring
        sd.play(signal, sample_rate)
        
        # Send the signal to Arduino for hardware control
        if self.arduino:
            try:
                # Scale signal to 0-255 for Arduino PWM
                scaled_signal = ((signal + 1) * 127.5).astype(int)  # Map [-1, 1] to [0, 255]
                for value in scaled_signal:
                    self.arduino.write(bytes([value]))
                    # Read feedback from hall sensor
                    if self.arduino.in_waiting > 0:
                        feedback = self.arduino.readline().decode().strip()
                        print(f"Hall Sensor Feedback: {feedback}")
                    time.sleep(1 / sample_rate)  # Match the sample rate timing
            except serial.SerialException as e:
                print(f"Error communicating with Arduino: {e}")
        
        sd.wait()

    def evolve_system(self, steps=100):
        """Evolve the full system including fields and particles."""
        for _ in range(steps):
            self.evolve_quantum_field_rk4()
            
            for i in range(self.resolution):
                for μ in range(4):
                    for ν in range(4):
                        self.em['F_munu'][i, μ, ν] = (self.em['A_mu'][(i+1)%self.resolution, ν] - 
                                                      self.em['A_mu'][(i-1)%self.resolution, ν] - 
                                                      self.em['A_mu'][(i+1)%self.resolution, μ] + 
                                                      self.em['A_mu'][(i-1)%self.resolution, μ]) / (2*self.dx)
            self.em['A_mu'] += np.random.normal(0, 1e-6, (self.resolution, 4))
            
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
            self.metric = self.compute_metric()
            self.christoffel = self.compute_christoffel_symbols_analytical()
            self.total_stress_energy()

        flux_signal = self.generate_flux_signal()
        self.activate_flux_capacitor(flux_signal)

    def move_charged_particles(self, dt):
        """Move charged particles along geodesics with field interactions."""
        for p in self.particles:
            state = np.concatenate([p['position'][:3], p['velocity'][:3]])
            t_range = np.linspace(0, dt, 2)
            trajectory = odeint(self.equations_of_motion, state, t_range)
            p['position'][:3] = trajectory[-1, :3]
            p['velocity'][:3] = trajectory[-1, 3:]
            p['path'].append(p['position'][:3].copy())

    def add_particle(self, position, velocity, charge):
        """Add a charged particle to the simulation."""
        particle = {
            'position': np.array(position, dtype=np.float32),
            'velocity': np.array(velocity, dtype=np.float32),
            'charge': charge,
            'path': [position[:3].copy()]
        }
        self.particles.append(particle)

    def visualize_unified_fields(self):
        """Visualize spacetime curvature, quantum fields, gravitational waves, and structure."""
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
        ax2.set_title('Quantum Field Densities (Schumann & Pythagorean Coupled)')
        ax2.legend()

        ax3 = fig.add_subplot(223)
        ax3.plot(self.fabric[:, 3], self.gw['plus'], label='+ Polarization')
        ax3.plot(self.fabric[:, 3], self.gw['cross'], label='× Polarization')
        ax3.set_title('Gravitational Waves (Schumann Modulated)')
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
        """Close the serial connection to the Arduino."""
        if self.arduino and self.arduino.is_open:
            self.arduino.close()
            print("Serial connection closed.")

if __name__ == "__main__":
    # Initialize the Flux Capacitor with default parameters
    # Update 'COM3' to your Arduino's serial port (e.g., '/dev/ttyUSB0' on Linux)
    sim = UnifiedSpacetimeSimulator(resolution=8, lambda_=1.0, kappa=0.1, charge_density=1e-15, serial_port='COM3')
    
    try:
        # Add particles to the simulation
        for v in sim.fabric:
            sim.add_particle(position=v, velocity=0.1 * sim.c * np.random.randn(4), charge=1.6e-19)
        
        # Evolve the system and activate the Flux Capacitor
        sim.evolve_system(steps=10)
        
        # Visualize the results
        sim.visualize_unified_fields()
    finally:
        # Ensure the serial connection is closed
        sim.close()
