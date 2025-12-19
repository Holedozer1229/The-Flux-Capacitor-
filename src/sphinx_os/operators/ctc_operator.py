"""
Formalized Closed Timelike Curve (CTC) Operator

This module provides a unified, mathematically rigorous implementation of CTC operators
for quantum-gravitational simulations, replacing ad-hoc implementations throughout the codebase.
"""

import numpy as np
import logging
from typing import Optional, Dict, Tuple, Union
from functools import lru_cache

logger = logging.getLogger(__name__)


class CTCOperator:
    """
    Formalized CTC operator for quantum-gravitational simulations.
    
    The CTC operator models non-local temporal effects through a phase shift operator
    that couples scalar fields, quantum states, and spacetime curvature via the J^6 potential.
    
    Mathematical form:
        CTC(t, φ, J6) = κ_ctc × sin(φt/τ) × J6 × B(z) × D(body_positions)
    
    where:
        - κ_ctc: CTC coupling constant
        - τ: CTC timescale parameter
        - φ: Scalar field value
        - J6: J^6 modulation term
        - B(z): AdS boundary factor
        - D: Three-body distance modulation
    """
    
    def __init__(
        self,
        tau: float = 1.0,
        kappa_ctc: float = 0.5,
        phase_factor: float = 0.1,
        wormhole_factor: float = 0.5,
        amplify_factor: float = 0.2,
        boundary_coupling: float = 0.1,
        j6_scale: float = 1e-30,
        epsilon: float = 1e-15
    ):
        """
        Initialize the CTC operator with physical parameters.
        
        Args:
            tau: CTC timescale (temporal period of oscillation)
            kappa_ctc: CTC coupling strength
            phase_factor: Phase modulation factor for quantum gates
            wormhole_factor: Wormhole node enhancement factor
            amplify_factor: Entanglement amplification factor
            boundary_coupling: AdS boundary coupling strength
            j6_scale: J^6 scaling factor for regularization
            epsilon: Small constant for numerical stability
        """
        # Validate parameters
        if not (0.1 <= tau <= 10.0):
            raise ValueError(f"tau must be in [0.1, 10.0], got {tau}")
        if not (0.1 <= kappa_ctc <= 2.0):
            raise ValueError(f"kappa_ctc must be in [0.1, 2.0], got {kappa_ctc}")
        if not (0.05 <= phase_factor <= 0.5):
            raise ValueError(f"phase_factor must be in [0.05, 0.5], got {phase_factor}")
        if not (0.2 <= wormhole_factor <= 1.0):
            raise ValueError(f"wormhole_factor must be in [0.2, 1.0], got {wormhole_factor}")
        if not (0.1 <= amplify_factor <= 0.4):
            raise ValueError(f"amplify_factor must be in [0.1, 0.4], got {amplify_factor}")
        
        self.tau = tau
        self.kappa_ctc = kappa_ctc
        self.phase_factor = phase_factor
        self.wormhole_factor = wormhole_factor
        self.amplify_factor = amplify_factor
        self.boundary_coupling = boundary_coupling
        self.j6_scale = j6_scale
        self.epsilon = epsilon
        
        logger.info(
            "Initialized CTCOperator: tau=%.3f, kappa_ctc=%.3f, phase_factor=%.3f, "
            "wormhole_factor=%.3f, amplify_factor=%.3f",
            tau, kappa_ctc, phase_factor, wormhole_factor, amplify_factor
        )
    
    def compute_boundary_factor(
        self,
        z: Union[float, np.ndarray],
        j6_modulation: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """
        Compute AdS boundary factor with non-linear J^6 coupling.
        
        Args:
            z: Normalized distance from AdS boundary (0 = boundary, 1 = bulk)
            j6_modulation: J^6 potential modulation term
        
        Returns:
            Boundary factor: exp(-α*z) * (1 + β*J6^6/(scale + ε))
        """
        nonlinear_j6 = j6_modulation**6 / (self.j6_scale + self.epsilon)
        boundary_factor = np.exp(-self.boundary_coupling * z) * (1 + 0.001 * nonlinear_j6)
        return boundary_factor
    
    def compute_ctc_term(
        self,
        t: float,
        phi: float,
        j6_modulation: float,
        z: Optional[float] = None,
        body_positions: Optional[list] = None
    ) -> float:
        """
        Compute the full CTC term for spacetime evolution.
        
        Args:
            t: Time coordinate
            phi: Scalar field value
            j6_modulation: J^6 potential modulation
            z: AdS boundary coordinate (optional, computed if not provided)
            body_positions: List of 3D body positions for distance modulation
        
        Returns:
            CTC term value
        """
        # Base CTC oscillation
        ctc_base = self.kappa_ctc * np.sin(phi * t / self.tau)
        
        # J^6 coupling
        ctc_term = ctc_base * j6_modulation
        
        # AdS boundary modulation
        if z is not None:
            boundary_factor = self.compute_boundary_factor(z, j6_modulation)
            ctc_term *= boundary_factor
        
        # Three-body distance modulation
        if body_positions:
            from ..utils.math_utils import compute_body_distance_sum
            dist_sum = compute_body_distance_sum(body_positions)
            ctc_term *= np.exp(-0.01 * dist_sum)
        
        return ctc_term
    
    def compute_phase_shift(
        self,
        phi: float,
        ctc_feedback: float,
        boundary_factor: float = 1.0,
        psi_abs_sq: float = 1.0,
        j4_abs: float = 0.0
    ) -> complex:
        """
        Compute quantum phase shift for Rydberg gates.
        
        Args:
            phi: Scalar field value
            ctc_feedback: CTC term feedback from evolution
            boundary_factor: AdS boundary factor
            psi_abs_sq: Quantum state intensity |ψ|²
            j4_abs: Electromagnetic field strength |J4|
        
        Returns:
            Complex phase factor exp(iθ)
        """
        # Non-linear boundary enhancement
        boundary_nonlinear = (psi_abs_sq * j4_abs**3)**6 / (self.j6_scale + self.epsilon)
        effective_boundary = boundary_factor * (1 + 0.001 * boundary_nonlinear)
        
        # Phase angle with CTC modulation
        phase_angle = (phi + self.phase_factor * ctc_feedback) * effective_boundary
        
        return np.exp(1j * phase_angle)
    
    def compute_wormhole_enhancement(
        self,
        ctc_feedback: float,
        golden_ratio: float = 1.618033988749895
    ) -> float:
        """
        Compute wormhole node enhancement factor.
        
        Args:
            ctc_feedback: CTC term feedback
            golden_ratio: Golden ratio φ = (1 + √5)/2
        
        Returns:
            Wormhole enhancement factor (clipped to [0.5, 5.0])
        """
        enhancement = golden_ratio * (1 + self.wormhole_factor * np.abs(ctc_feedback))
        return np.clip(enhancement, 0.5, 5.0)
    
    def compute_entanglement_amplification(
        self,
        expectation_value: float,
        ctc_feedback: float,
        amplify: bool = False
    ) -> float:
        """
        Compute entanglement amplification via CTC effects.
        
        Args:
            expectation_value: Base expectation value (e.g., CHSH, MABK, GHZ)
            ctc_feedback: CTC term feedback
            amplify: Whether to apply amplification
        
        Returns:
            Amplified expectation value
        """
        if not amplify:
            return expectation_value
        
        amplification = 1 + self.amplify_factor * np.abs(ctc_feedback)
        return expectation_value * amplification
    
    def compute_m_shift(
        self,
        u: float,
        v: float,
        phi: float,
        t: float,
        oam: float = 0.0,
        order: int = 2
    ) -> float:
        """
        Compute CTC-modulated m_shift function for Möbius geometry.
        
        Args:
            u: First coordinate
            v: Second coordinate
            phi: Scalar field value
            t: Time coordinate
            oam: Orbital angular momentum
            order: Polynomial order for higher-order corrections
        
        Returns:
            CTC-modulated m_shift value
        """
        ctc_base = self.kappa_ctc * np.sin(phi * t / self.tau)
        
        # Base m_shift (using Euler's number e ≈ 2.72)
        m_shift = 2.72
        
        # Linear correction
        m_shift *= (1 + 0.01 * ctc_base * (1 + 0.002 * oam))
        
        # Higher-order corrections
        if order >= 2:
            m_shift *= (1 + 0.001 * ctc_base**2)
        if order >= 4:
            m_shift *= (1 + 0.0001 * ctc_base**4)
        
        return m_shift
    
    def apply_non_local_gate_phases(
        self,
        wires: list,
        phi: float,
        t: float,
        oam: float = 0.0,
        order: int = 4
    ) -> Dict[str, list]:
        """
        Compute phase angles for non-local quantum gates.
        
        Args:
            wires: List of qubit wire indices
            phi: Scalar field value
            t: Time coordinate
            oam: Orbital angular momentum
            order: Maximum polynomial order for corrections
        
        Returns:
            Dictionary with gate types and their phase angles
        """
        ctc_base = self.kappa_ctc * np.sin(phi * t / self.tau)
        
        # Single-qubit phases with higher-order corrections
        single_qubit_phases = []
        for _ in wires:
            phase = ctc_base * (1 + 0.002 * oam)
            if order >= 2:
                phase *= (1 + 0.001 * ctc_base**2)
            if order >= 4:
                phase *= (1 + 0.0001 * ctc_base**4)
            single_qubit_phases.append(phase)
        
        # Two-qubit controlled phases
        controlled_phases = []
        if len(wires) > 1:
            for i in range(len(wires) - 1):
                phase = ctc_base * 0.2 * (1 + 0.002 * oam)
                controlled_phases.append(phase)
        
        # Multi-qubit higher-order phases
        multi_qubit_phases = []
        if len(wires) >= 3:
            phase = ctc_base * 0.1 * (1 + 0.002 * oam**2)
            multi_qubit_phases.append(phase)
        if len(wires) >= 4:
            phase = ctc_base * 0.05 * (1 + 0.002 * oam)
            multi_qubit_phases.append(phase)
        
        return {
            'single_qubit': single_qubit_phases,
            'controlled': controlled_phases,
            'multi_qubit': multi_qubit_phases
        }
    
    def to_dict(self) -> Dict[str, float]:
        """Export operator parameters as dictionary."""
        return {
            'tau': self.tau,
            'kappa_ctc': self.kappa_ctc,
            'ctc_phase_factor': self.phase_factor,
            'ctc_wormhole_factor': self.wormhole_factor,
            'ctc_amplify_factor': self.amplify_factor,
            'boundary_coupling': self.boundary_coupling
        }
    
    @classmethod
    def from_dict(cls, params: Dict[str, float]) -> 'CTCOperator':
        """Create operator from parameter dictionary."""
        return cls(
            tau=params.get('tau', 1.0),
            kappa_ctc=params.get('kappa_ctc', 0.5),
            phase_factor=params.get('ctc_phase_factor', 0.1),
            wormhole_factor=params.get('ctc_wormhole_factor', 0.5),
            amplify_factor=params.get('ctc_amplify_factor', 0.2),
            boundary_coupling=params.get('boundary_coupling', 0.1)
        )
    
    def __repr__(self) -> str:
        return (
            f"CTCOperator(tau={self.tau:.3f}, kappa_ctc={self.kappa_ctc:.3f}, "
            f"phase_factor={self.phase_factor:.3f}, wormhole_factor={self.wormhole_factor:.3f}, "
            f"amplify_factor={self.amplify_factor:.3f})"
        )


# Convenience function for backward compatibility
def compute_ctc_term(
    t: float,
    phi: float,
    j6_modulation: float,
    ctc_params: Optional[Dict[str, float]] = None,
    z: Optional[float] = None,
    body_positions: Optional[list] = None
) -> float:
    """
    Compute CTC term using the formalized operator.
    
    This function provides backward compatibility with the previous ad-hoc implementation.
    
    Args:
        t: Time coordinate
        phi: Scalar field value
        j6_modulation: J^6 potential modulation
        ctc_params: Optional dictionary of CTC parameters
        z: AdS boundary coordinate
        body_positions: List of 3D body positions
    
    Returns:
        CTC term value
    """
    if ctc_params:
        operator = CTCOperator.from_dict(ctc_params)
    else:
        operator = CTCOperator()
    
    return operator.compute_ctc_term(t, phi, j6_modulation, z, body_positions)
