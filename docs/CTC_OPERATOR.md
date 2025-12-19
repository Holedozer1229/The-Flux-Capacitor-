# Formalized CTC Operator Documentation

## Overview

The `CTCOperator` class provides a unified, mathematically rigorous implementation of Closed Timelike Curve (CTC) operators for quantum-gravitational simulations in The Flux Capacitor framework.

## Mathematical Formulation

The CTC operator models non-local temporal effects through a phase shift operator that couples scalar fields, quantum states, and spacetime curvature via the J^6 potential:

```
CTC(t, φ, J6) = κ_ctc × sin(φt/τ) × J6 × B(z) × D(body_positions)
```

Where:
- **κ_ctc**: CTC coupling constant (strength of temporal non-locality)
- **τ**: CTC timescale parameter (temporal period of oscillation)
- **φ**: Scalar field value (Nugget field)
- **J6**: J^6 modulation term (unified potential)
- **B(z)**: AdS boundary factor with non-linear J^6 coupling
- **D**: Three-body distance modulation factor

## Key Features

### 1. Parameter Validation
All physical parameters are validated to ensure they lie within physically meaningful ranges:

- `tau`: [0.1, 10.0] - CTC timescale
- `kappa_ctc`: [0.1, 2.0] - CTC coupling strength
- `phase_factor`: [0.05, 0.5] - Phase modulation factor
- `wormhole_factor`: [0.2, 1.0] - Wormhole enhancement factor
- `amplify_factor`: [0.1, 0.4] - Entanglement amplification factor

### 2. Core Methods

#### `compute_ctc_term(t, phi, j6_modulation, z=None, body_positions=None)`
Computes the full CTC term for spacetime evolution, including:
- Base CTC oscillation: κ × sin(φt/τ)
- J^6 coupling
- AdS boundary modulation
- Three-body distance modulation

#### `compute_phase_shift(phi, ctc_feedback, boundary_factor, psi_abs_sq, j4_abs)`
Computes quantum phase shift for Rydberg gates:
- Non-linear boundary enhancement
- CTC-modulated phase angle
- Returns complex phase factor exp(iθ)

#### `compute_wormhole_enhancement(ctc_feedback, golden_ratio)`
Computes wormhole node enhancement using the golden ratio:
- Enhancement = φ × (1 + factor × |CTC|)
- Clipped to [0.5, 5.0] for stability

#### `compute_entanglement_amplification(expectation_value, ctc_feedback, amplify)`
Amplifies entanglement measures (CHSH, MABK, GHZ) via CTC effects:
- Amplification = 1 + factor × |CTC|
- Optional amplification flag

#### `compute_m_shift(u, v, phi, t, oam, order)`
Computes CTC-modulated m_shift for Möbius geometry:
- Base: e ≈ 2.72
- Linear corrections with OAM coupling
- Higher-order corrections (quadratic, quartic)

#### `apply_non_local_gate_phases(wires, phi, t, oam, order)`
Computes phase angles for non-local quantum gates:
- Single-qubit phases with higher-order corrections
- Two-qubit controlled phases
- Multi-qubit higher-order phases

### 3. Serialization
- `to_dict()`: Export parameters as dictionary
- `from_dict(params)`: Create operator from dictionary
- Backward compatibility with previous implementations

## Usage Examples

### Basic Usage

```python
from sphinx_os.operators import CTCOperator

# Initialize with default parameters
operator = CTCOperator()

# Compute CTC term
ctc_term = operator.compute_ctc_term(
    t=1.0,
    phi=0.5,
    j6_modulation=1.0
)
```

### With Body Positions

```python
import numpy as np

# Three-body system
body_positions = [
    np.array([0.0, 0.0, 0.0]),
    np.array([1.0, 0.0, 0.0]),
    np.array([0.0, 1.0, 0.0])
]

ctc_term = operator.compute_ctc_term(
    t=1.0,
    phi=0.5,
    j6_modulation=1.0,
    body_positions=body_positions
)
```

### Quantum Phase Shifts

```python
# Compute phase shift for quantum gates
phase = operator.compute_phase_shift(
    phi=0.5,
    ctc_feedback=0.1,
    boundary_factor=1.0,
    psi_abs_sq=1.0,
    j4_abs=0.0
)

# Apply to quantum state
state *= phase
```

### Entanglement Amplification

```python
# Compute CHSH violation
chsh_base = 2.828

# Amplify with CTC effects
chsh_amplified = operator.compute_entanglement_amplification(
    chsh_base,
    ctc_feedback=0.5,
    amplify=True
)
```

### Custom Parameters

```python
# Create operator with custom parameters
operator = CTCOperator(
    tau=1.5,
    kappa_ctc=0.7,
    phase_factor=0.2,
    wormhole_factor=0.6,
    amplify_factor=0.25
)

# Export parameters
params = operator.to_dict()

# Recreate from parameters
operator2 = CTCOperator.from_dict(params)
```

### Backward Compatibility

```python
from sphinx_os.operators import compute_ctc_term

# Use convenience function (maintains backward compatibility)
ctc_term = compute_ctc_term(
    t=1.0,
    phi=0.5,
    j6_modulation=1.0,
    ctc_params={'tau': 1.5, 'kappa_ctc': 0.7}
)
```

## Integration with Existing Code

The formalized CTC operator can be integrated into existing simulations by:

1. **Replacing ad-hoc CTC calculations** with `operator.compute_ctc_term()`
2. **Using standardized phase shifts** with `operator.compute_phase_shift()`
3. **Applying consistent amplification** with `operator.compute_entanglement_amplification()`
4. **Computing gate phases** with `operator.apply_non_local_gate_phases()`

### Example Integration

```python
# Old code (ad-hoc)
ctc_term = kappa_ctc * np.sin(phi * t / tau) * j6_modulation

# New code (formalized)
operator = CTCOperator(tau=tau, kappa_ctc=kappa_ctc)
ctc_term = operator.compute_ctc_term(t, phi, j6_modulation)
```

## Performance Considerations

- The operator uses numpy for vectorized operations
- AdS boundary factors are computed efficiently
- Distance calculations leverage the cached `compute_body_distance_sum()` function
- Parameter validation occurs at initialization, not during computation

## Testing

Comprehensive tests are provided in `tests/test_ctc_operator.py`:

```bash
pytest tests/test_ctc_operator.py -v
```

Tests cover:
- Parameter validation
- All core methods
- Boundary conditions
- Edge cases
- Serialization/deserialization
- Backward compatibility

## References

- **CTC Theory**: Closed timelike curves in general relativity
- **AdS/CFT**: Anti-de Sitter/Conformal Field Theory correspondence
- **J^6 Coupling**: Unified potential coupling scalar fields, quantum states, and curvature
- **Three-Body Problem**: Classical three-body gravitational dynamics

## Future Enhancements

Potential future improvements:
1. GPU acceleration for large-scale simulations
2. Adaptive parameter optimization
3. Higher-order CTC corrections (beyond quartic)
4. Integration with quantum error correction
5. Holographic CTC encoding
