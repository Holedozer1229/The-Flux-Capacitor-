"""Tests for the formalized CTC operator."""

import numpy as np
import pytest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from sphinx_os.operators.ctc_operator import CTCOperator, compute_ctc_term


def test_ctc_operator_initialization():
    """Test CTC operator initialization with valid parameters."""
    operator = CTCOperator(
        tau=1.0,
        kappa_ctc=0.5,
        phase_factor=0.1,
        wormhole_factor=0.5,
        amplify_factor=0.2
    )
    
    assert operator.tau == 1.0
    assert operator.kappa_ctc == 0.5
    assert operator.phase_factor == 0.1
    assert operator.wormhole_factor == 0.5
    assert operator.amplify_factor == 0.2


def test_ctc_operator_validation():
    """Test parameter validation."""
    # Test invalid tau
    with pytest.raises(ValueError):
        CTCOperator(tau=0.05)
    
    # Test invalid kappa_ctc
    with pytest.raises(ValueError):
        CTCOperator(kappa_ctc=0.05)
    
    # Test invalid phase_factor
    with pytest.raises(ValueError):
        CTCOperator(phase_factor=0.01)
    
    # Test invalid wormhole_factor
    with pytest.raises(ValueError):
        CTCOperator(wormhole_factor=0.1)
    
    # Test invalid amplify_factor
    with pytest.raises(ValueError):
        CTCOperator(amplify_factor=0.05)


def test_compute_boundary_factor():
    """Test AdS boundary factor computation."""
    operator = CTCOperator()
    
    # Test at boundary (z=0)
    boundary_z0 = operator.compute_boundary_factor(0.0, 1.0)
    assert np.isfinite(boundary_z0)
    assert boundary_z0 > 0
    
    # Test in bulk (z=1)
    boundary_z1 = operator.compute_boundary_factor(1.0, 1.0)
    assert np.isfinite(boundary_z1)
    assert boundary_z1 > 0
    assert boundary_z1 < boundary_z0  # Boundary should be larger


def test_compute_ctc_term():
    """Test CTC term computation."""
    operator = CTCOperator(tau=1.0, kappa_ctc=0.5)
    
    # Basic test
    ctc_term = operator.compute_ctc_term(t=1.0, phi=0.5, j6_modulation=1.0)
    assert np.isfinite(ctc_term)
    
    # Test with boundary coordinate
    ctc_term_with_z = operator.compute_ctc_term(t=1.0, phi=0.5, j6_modulation=1.0, z=0.5)
    assert np.isfinite(ctc_term_with_z)
    
    # Test with body positions
    body_positions = [
        np.array([0.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0])
    ]
    ctc_term_with_bodies = operator.compute_ctc_term(
        t=1.0, phi=0.5, j6_modulation=1.0, body_positions=body_positions
    )
    assert np.isfinite(ctc_term_with_bodies)


def test_compute_phase_shift():
    """Test quantum phase shift computation."""
    operator = CTCOperator()
    
    phase = operator.compute_phase_shift(
        phi=0.5,
        ctc_feedback=0.1,
        boundary_factor=1.0,
        psi_abs_sq=1.0,
        j4_abs=0.0
    )
    
    assert np.isfinite(phase)
    assert np.abs(phase) > 0
    assert np.abs(np.abs(phase) - 1.0) < 1e-10  # Should be unit magnitude


def test_compute_wormhole_enhancement():
    """Test wormhole enhancement factor."""
    operator = CTCOperator(wormhole_factor=0.5)
    
    # Test with no feedback
    enhancement_zero = operator.compute_wormhole_enhancement(0.0)
    assert np.isfinite(enhancement_zero)
    assert 0.5 <= enhancement_zero <= 5.0
    
    # Test with positive feedback
    enhancement_pos = operator.compute_wormhole_enhancement(1.0)
    assert np.isfinite(enhancement_pos)
    assert enhancement_pos > enhancement_zero
    assert 0.5 <= enhancement_pos <= 5.0


def test_compute_entanglement_amplification():
    """Test entanglement amplification."""
    operator = CTCOperator(amplify_factor=0.2)
    
    # Test without amplification
    result_no_amp = operator.compute_entanglement_amplification(2.828, 0.5, amplify=False)
    assert result_no_amp == 2.828
    
    # Test with amplification
    result_with_amp = operator.compute_entanglement_amplification(2.828, 0.5, amplify=True)
    assert result_with_amp > 2.828
    assert np.isfinite(result_with_amp)


def test_compute_m_shift():
    """Test m_shift computation for Möbius geometry."""
    operator = CTCOperator()
    
    # Test with different orders
    m_shift_order2 = operator.compute_m_shift(0.5, 0.5, 1.0, 0.5, oam=0.0, order=2)
    assert np.isfinite(m_shift_order2)
    assert m_shift_order2 > 0
    
    m_shift_order4 = operator.compute_m_shift(0.5, 0.5, 1.0, 0.5, oam=0.0, order=4)
    assert np.isfinite(m_shift_order4)
    assert m_shift_order4 > 0


def test_apply_non_local_gate_phases():
    """Test non-local gate phase computation."""
    operator = CTCOperator()
    
    wires = [0, 1, 2, 3]
    phases = operator.apply_non_local_gate_phases(wires, phi=1.0, t=0.5, oam=0.0, order=4)
    
    assert 'single_qubit' in phases
    assert 'controlled' in phases
    assert 'multi_qubit' in phases
    
    assert len(phases['single_qubit']) == len(wires)
    assert len(phases['controlled']) == len(wires) - 1
    assert len(phases['multi_qubit']) >= 1
    
    # Check all phases are finite
    for phase_list in phases.values():
        for phase in phase_list:
            assert np.isfinite(phase)


def test_to_dict_and_from_dict():
    """Test serialization and deserialization."""
    operator1 = CTCOperator(
        tau=1.5,
        kappa_ctc=0.7,
        phase_factor=0.2,
        wormhole_factor=0.6,
        amplify_factor=0.25
    )
    
    # Convert to dict
    params = operator1.to_dict()
    assert isinstance(params, dict)
    assert 'tau' in params
    assert params['tau'] == 1.5
    
    # Recreate from dict
    operator2 = CTCOperator.from_dict(params)
    assert operator2.tau == operator1.tau
    assert operator2.kappa_ctc == operator1.kappa_ctc
    assert operator2.phase_factor == operator1.phase_factor
    assert operator2.wormhole_factor == operator1.wormhole_factor
    assert operator2.amplify_factor == operator1.amplify_factor


def test_backward_compatibility():
    """Test backward compatibility function."""
    # Test without parameters
    ctc_term1 = compute_ctc_term(t=1.0, phi=0.5, j6_modulation=1.0)
    assert np.isfinite(ctc_term1)
    
    # Test with parameters
    ctc_params = {
        'tau': 1.5,
        'kappa_ctc': 0.7,
        'ctc_phase_factor': 0.2
    }
    ctc_term2 = compute_ctc_term(t=1.0, phi=0.5, j6_modulation=1.0, ctc_params=ctc_params)
    assert np.isfinite(ctc_term2)


def test_repr():
    """Test string representation."""
    operator = CTCOperator(tau=1.0, kappa_ctc=0.5)
    repr_str = repr(operator)
    assert 'CTCOperator' in repr_str
    assert 'tau=1.000' in repr_str
    assert 'kappa_ctc=0.500' in repr_str


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
