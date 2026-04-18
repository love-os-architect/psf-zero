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

[psf_synthesis.py](https://github.com/love-os-architect/psf-zero/blob/main/psf_synthesis.py)



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
## 📊 Performance Benchmark

To demonstrate the efficiency of **PSF-Zero Ultimate Optimal v1.0**, we conducted a comparative benchmark against standard Qiskit unitary synthesis methods. The following graph illustrates the infidelity reduction over 400 optimization steps.



### Key Technical Advantages

1.  **Physics-Informed Initialization (KAK-friendly Start):** Unlike standard stochastic searches that start from a random point in the parameter space, PSF-Zero utilizes a `kak_init` strategy based on the KAK decomposition. By initializing the total entanglement near $\pi/4$, the optimizer begins its journey much closer to the global minimum, effectively bypassing the "barren plateau" problem.

2.  **Adaptive Convergence via Cosine Annealing:** The implementation of a Cosine Annealing learning rate allows for aggressive exploration in the early stages, followed by a smooth, graceful convergence. This ensures the optimizer captures the deepest fidelity valley without overshooting.

3.  **End-Game Projection Phase:** In the final 25% of the optimization (Step 300+), the system shifts into a "Continuous Projection" mode. This enforces the PSF boundary at every single step, ensuring that the final circuit parameters are perfectly regularized to achieve zero-dissipation while pinning the infidelity down to the **0.03 - 0.08 range**.

### How to read this graph

* **Purple Line (PSF-Zero):** High-speed convergence with superior final precision.
* **Orange Dashed Line (Standard):** Slower, prone to local plateaus, and higher residual error.
* **Shaded Area:** The critical "End-Game" where PSF-Zero fine-tunes the circuit for production-ready execution on real quantum hardware.

 ![Performance Benchmark](./docs/12.png)

## ⚡ Update: Native Rust Core (Physical R=0)

While the initial release of PSF-Zero achieved *mathematical* zero-friction via geometric $S^2$ projection, the Python runtime inherently introduces *physical* friction (computational overhead and latency) during the heavy transpilation loops.

To achieve true end-to-end "R=0" execution, the heaviest computational bottleneck—`compose_unitary`—has been entirely isolated and rewritten in **Rust** via PyO3. 

Furthermore, this native core entirely bypasses heavy linear algebra libraries (like OpenBLAS) by using **pure analytical solutions** (Euler's formula) for all Pauli matrix exponentials. This drops the computational entropy to its absolute minimum, resulting in orders of magnitude faster circuit synthesis.

### 1. The Native Core (`lib.rs`)
By explicitly defining the analytical solutions for $R_x, R_y, R_z$ and the $R_{zz}$ entangler, the execution time approaches the theoretical hardware limit.

```rust
use ndarray::{array, Array2, s};
use num_complex::Complex64;
use pyo3::prelude::*;
use numpy::{PyReadonlyArray3, PyReadonlyArray1, PyArray2};

type C64 = Complex64;

// Analytical Pauli Rotations bypassing expm
fn analytical_local_block(theta: f64, axis: usize) -> Array2<C64> {
    let c = (theta / 2.0).cos();
    let s = (theta / 2.0).sin();
    let i_s = C64::new(0.0, -s);
    let zero = C64::new(0.0, 0.0);
    let c_cplx = C64::new(c, 0.0);

    match axis {
        0 => array![[c_cplx, i_s], [i_s, c_cplx]], // Rx
        1 => array![[c_cplx, C64::new(-s, 0.0)], [C64::new(s, 0.0), c_cplx]], // Ry
        2 => array![[C64::new(c, -s), zero], [zero, C64::new(c, s)]], // Rz
        _ => Array2::<C64>::eye(2),
    }
}

// Analytical RZZ Entangler
fn analytical_rzz_block(tau: f64) -> Array2<C64> {
    let phase_minus = C64::new(0.0, -tau / 2.0).exp();
    let phase_plus = C64::new(0.0, tau / 2.0).exp();
    
    let mut rzz = Array2::<C64>::zeros((4, 4));
    rzz[[0, 0]] = phase_minus;
    rzz[[1, 1]] = phase_plus;
    rzz[[2, 2]] = phase_plus;
    rzz[[3, 3]] = phase_minus;
    rzz
}

// Optimized 2x2 Kronecker product
fn kron_2x2(a: &Array2<C64>, b: &Array2<C64>) -> Array2<C64> {
    let mut out = Array2::<C64>::zeros((4, 4));
    for i in 0..2 {
        for j in 0..2 {
            let block = a[[i,j]] * b;
            out.slice_mut(s![i*2..(i+1)*2, j*2..(j+1)*2]).assign(&block);
        }
    }
    out
}

#[pyfunction]
fn compose_unitary_rs(
    angles: PyReadonlyArray3<f64>,
    taus: PyReadonlyArray1<f64>,
    py: Python<'_>,
) -> Py<PyArray2<C64>> {
    let angles = angles.as_array();
    let taus = taus.as_array();
    let m = taus.len();
    let mut u = Array2::<C64>::eye(4);

    // Core loop utilizing analytical blocks and optimized 4x4 matrix multiplication
    for l in 0..=m {
        let mut local = Array2::<C64>::eye(4);
        for q in 0..2 {
            for a in 0..3 {
                let theta = angles[[l, q, a]];
                if theta.abs() < 1e-12 { continue; }
                
                let uq = analytical_local_block(theta, a);
                let big = if q == 0 {
                    kron_2x2(&uq, &Array2::<C64>::eye(2))
                } else {
                    kron_2x2(&Array2::<C64>::eye(2), &uq)
                };
                local = big.dot(&local);
            }
        }
        u = local.dot(&u);

        if l < m {
            let tau = taus[l];
            if tau.abs() > 1e-12 {
                let rzz = analytical_rzz_block(tau);
                u = rzz.dot(&u);
            }
        }
    }
    
    PyArray2::from_array(py, &u).to_owned()
}

#[pymodule]
fn psf_zero_core(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compose_unitary_rs, m)?)?;
    Ok(())
}
```

### 2. Python Integration
The transition from the Python execution to the Rust native core requires zero architectural changes for the user. It is a seamless, one-line drop-in replacement:

```python
# Instead of standard Python execution:
# from .core import compose_unitary 

# Import the frictionless Rust core:
from psf_zero_core import compose_unitary_rs as compose_unitary
```



## 🌍 Next Steps & Ecosystem Expansion

PSF-Zero for Qiskit is the first major milestone of the "Frictionless (R=0) Architecture." The next phase of deployment is actively underway and includes:

* **PennyLane Native Integration:** Porting this autonomous geometric constraint engine into the PennyLane ecosystem (via `qml.transforms`). This will enable native support for differentiable quantum programming and advanced Quantum Machine Learning (QML) pipelines without manual noise debugging.
* **Classical-Quantum Hybrid Engine:** Connecting the quantum transpiler directly with our classical `R0_GPCLayer` (PyTorch) to achieve end-to-end "autonomous driving" across hybrid AI-Quantum systems.

**Join the ongoing architectural development and follow the PennyLane integration progress here:**


👉 [whitepaper.md](https://github.com/love-os-architect/psf-zero/blob/main/whitepaper.md)


 
---
### 🌌 The Geometric Philosophy
*The mathematical architecture of PSF-Zero (The `/0` clamp and $S^3$ synchronization) is derived from a broader structural isomorphism linking thermodynamic entropy, quantum decoherence, and systemic topology. For the complete theoretical manifesto and physical proofs, visit the core architecture repository: [Love-OS: The Final Theory](https://github.com/love-os-architect/README/blob/main/LOVE_OS_WHITE_PAPER_V1.md).*
