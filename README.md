# PSF-Zero: Projective Spherical Filtering for Quantum Control

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Qiskit Ecosystem](https://img.shields.io/badge/Qiskit-Ecosystem-purple.svg)](https://github.com/qiskit/ecosystem)

**PSF-Zero** is a manifold-aware geometric optimizer and Qiskit `TransformationPass` designed to synthesize highly robust, low-dissipation 2-qubit unitary circuits. 

By applying Projective Spherical Filtering (the `/0` clamp) and restricting parameter updates to minimal arcs on the $S^3 \cong SU(2)$ manifold, PSF-Zero inherently minimizes pulse dissipation (L1/TV norms) while avoiding the catastrophic "unwinding" and barren plateaus common in classical Euclidean optimizers.

## 🚀 Key Features

- **Geometric Step Saturation (`/0` Clamp):** Dynamically clips optimization steps based on curvature-aware trust regions, completely preventing rotational overshoot.
- **Analytic Subgradients:** Replaces slow finite-difference loops with $O(1)$ analytic subgradients for L1 (dissipation) and Total Variation (smoothness) penalties, drastically reducing transpilation time.
- **Manifold Adam Momentum:** Preserves 1st and 2nd order moments in the Lie algebra tangent space, ensuring rapid convergence out of barren plateaus.
- **Native Qiskit Integration:** Drops seamlessly into any existing Qiskit `PassManager` to automatically optimize `UnitaryGate` nodes into native entanglers (`RZZ`) and local rotations.

## 📦 Installation

Clone the repository and install via pip:

```bash
git clone https://github.com/YOUR_USERNAME/psf-zero.git
cd psf-zero
pip install -e .
```
*(Dependencies: `numpy`, `scipy`, `qiskit`)*

## 💻 Quickstart

PSF-Zero acts as a standard Qiskit transpiler pass. Simply add `PSFGateSynthesis` to your pass manager to automatically optimize all 2-qubit unitaries in your DAG.

```python
import numpy as np
from qiskit import QuantumCircuit
from qiskit.transpiler import PassManager
from qiskit.circuit.library import UnitaryGate
from psf_synthesis import PSFHyper, PSFGateSynthesis

# 1. Create a circuit with a target 2Q Unitary
qc = QuantumCircuit(2)
random_matrix = np.random.rand(4, 4) + 1j * np.random.rand(4, 4)
Q, _ = np.linalg.qr(random_matrix) # Generate a random SU(4) matrix
qc.append(UnitaryGate(Q), [0, 1])

# 2. Configure PSF-Zero Hyperparameters
hyper = PSFHyper(
    m=3,                 # Number of entangling RZZ gates
    iters=150,           # Optimization iterations
    lr=0.25,             # Base learning rate
    alpha_proj=1e-2,     # /0 geometric regularization strength
    beta_H=5e-3          # L1 Pulse dissipation penalty
)

# 3. Transpile using the PSF-Zero Synthesis Pass
pm = PassManager([PSFGateSynthesis(hyper)])
optimized_qc = pm.run(qc)

print("Synthesized Low-Dissipation Circuit:")
print(optimized_qc.draw())
```

## 📊 Performance & Calibration

PSF-Zero effectively solves the trade-off between Gate Fidelity ($F_{avg}$) and Control Dissipation. Check the included Jupyter Notebook (`01_psf_gate_calibration.ipynb`) to visualize the learning curve and convergence properties of the `PSFHybridSynthesizer`.

## 📜 Citation

If you use PSF-Zero in your quantum research or circuit optimization pipeline, please cite this repository using the included `CITATION.cff` or the following BibTeX:

```bibtex
@software{psf_zero_2026,
  author = {The Architect},
  title = {PSF-Zero: Zero-Dissipation Quantum Control Kernel for Qiskit},
  year = {2026},
  url = {https://github.com/YOUR_USERNAME/psf-zero},
  license = {AGPL-3.0}
}
```

---
### 🌌 The Geometric Philosophy
*The mathematical architecture of PSF-Zero (The `/0` clamp and $S^3$ synchronization) is derived from a broader structural isomorphism linking thermodynamic entropy, quantum decoherence, and systemic topology. For the complete theoretical manifesto and physical proofs, visit the core architecture repository: [Love-OS: The Final Theory](https://github.com/love-os-architect/README/blob/main/LOVE_OS_WHITE_PAPER_V1.md).*
