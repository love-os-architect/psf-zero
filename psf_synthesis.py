from __future__ import annotations
import numpy as np
import dataclasses
from dataclasses import dataclass
from typing import Optional
from scipy.linalg import expm, logm

from qiskit import QuantumCircuit
from qiskit.circuit.library import RZZGate, RXGate, RYGate, RZGate
from qiskit.transpiler.passes.synthesis.plugin import UnitarySynthesisPlugin

# =========================================================
# Pauli matrices
# =========================================================
I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

# =========================================================
# /0 Projective regularization
# =========================================================
def projective_reg(x: np.ndarray) -> np.ndarray:
    return x / np.sqrt(1.0 + x**2)

def projective_grad(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + x**2) ** 1.5

# =========================================================
# Local + RZZ blocks
# =========================================================
def local_block(angles: np.ndarray) -> np.ndarray:
    U = np.eye(4, dtype=complex)
    paulis = [X, Y, Z]
    for q in range(2):
        for a in range(3):
            theta = angles[q, a]
            Uq = expm(-1j * theta / 2 * paulis[a])
            U = np.kron(Uq, I) @ U if q == 0 else np.kron(I, Uq) @ U
    return U

def rzz_block(tau: float) -> np.ndarray:
    return expm(-1j * tau / 2 * np.kron(Z, Z))

def compose_unitary(angles: np.ndarray, taus: np.ndarray) -> np.ndarray:
    U = np.eye(4, dtype=complex)
    m = taus.shape[0]
    for l in range(m):
        U = local_block(angles[l]) @ U
        U = rzz_block(taus[l]) @ U
    return local_block(angles[-1]) @ U

# =========================================================
# Metrics
# =========================================================
def F_avg(U: np.ndarray, V: np.ndarray) -> float:
    d = U.shape[0]
    return float((np.abs(np.trace(U.conj().T @ V))**2 + d) / (d * (d + 1)))

def geodesic_loss(U: np.ndarray, U_target: np.ndarray) -> float:
    delta = logm(U_target.conj().T @ U)
    return float(np.real(np.trace(delta.conj().T @ delta)))

# =========================================================
# Parameter-shift gradients
# =========================================================
def parameter_shift_grad(angles: np.ndarray, taus: np.ndarray, U_target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    m = taus.shape[0]
    g_a = np.zeros_like(angles)
    g_t = np.zeros_like(taus)
    shift = np.pi / 2.0

    for l in range(m + 1):
        for q in range(2):
            for a in range(3):
                Ap = angles.copy()
                Am = angles.copy()
                Ap[l, q, a] += shift
                Am[l, q, a] -= shift
                g_a[l, q, a] = -0.5 * (
                    F_avg(compose_unitary(Ap, taus), U_target)
                    - F_avg(compose_unitary(Am, taus), U_target)
                )

    for l in range(m):
        tp = taus.copy()
        tm = taus.copy()
        tp[l] += shift
        tm[l] -= shift
        g_t[l] = -0.5 * (
            F_avg(compose_unitary(angles, tp), U_target)
            - F_avg(compose_unitary(angles, tm), U_target)
        )

    return g_a, g_t

# =========================================================
# Hyperparameters
# =========================================================
@dataclass
class PSFHyper:
    m: int = 3
    iters: int = 120
    lr: float = 0.06
    alpha_proj: float = 0.02
    beta_L1: float = 5e-4
    beta_TV: float = 5e-4
    beta_geo: float = 5e-3
    seeds: int = 3
    proj_every: int = 3
    tau_cal: Optional[np.ndarray] = None

# =========================================================
# Final Synthesizer 1.2
# =========================================================
class PSFHybridSynthesizerFinal:
    def __init__(self, hyper: PSFHyper):
        self.hyper = hyper
        self.angles = None
        self.taus = None

    def run(self, U_target: np.ndarray) -> float:
        global_best = np.inf

        for seed in range(self.hyper.seeds):
            rng = np.random.default_rng(seed)
            angles = rng.normal(scale=0.25, size=(self.hyper.m + 1, 2, 3))
            taus   = rng.normal(scale=0.2, size=(self.hyper.m,))

            seed_best = np.inf
            seed_params = None

            for step in range(self.hyper.iters):
                U = compose_unitary(angles, taus)
                diff = angles[:-1] - angles[1:]
                
                loss = (
                    1.0 - F_avg(U, U_target)
                    + self.hyper.beta_geo * geodesic_loss(U, U_target)
                    + self.hyper.beta_L1 * np.sum(np.abs(angles))
                    + self.hyper.beta_TV * np.sum(np.abs(diff))
                )

                if self.hyper.tau_cal is not None:
                    loss += self.hyper.beta_geo * np.sum((taus - self.hyper.tau_cal)**2)

                if loss < seed_best:
                    seed_best = loss
                    seed_params = (angles.copy(), taus.copy())

                g_a, g_t = parameter_shift_grad(angles, taus, U_target)
                g_a += self.hyper.beta_L1 * np.sign(angles)

                g_a[:-1] += self.hyper.beta_TV * np.sign(diff)
                g_a[1:]  -= self.hyper.beta_TV * np.sign(diff)

                if self.hyper.tau_cal is not None:
                    g_t += 2.0 * self.hyper.beta_geo * (taus - self.hyper.tau_cal)

                # projective damping (tunable)
                g_a *= self.hyper.alpha_proj * projective_grad(angles)
                g_t *= self.hyper.alpha_proj * projective_grad(taus)

                angles -= self.hyper.lr * g_a
                taus   -= self.hyper.lr * g_t

                if step % self.hyper.proj_every == 0:
                    angles = projective_reg(angles)
                    taus   = projective_reg(taus)

            if seed_best < global_best:
                global_best = seed_best
                if seed_params is not None:
                    self.angles, self.taus = seed_params

        return global_best

# =========================================================
# Qiskit Translation (as_qiskit)
# =========================================================
def params_to_qiskit(angles: np.ndarray, taus: np.ndarray) -> QuantumCircuit:
    """Convert the optimized angles and taus into a Qiskit QuantumCircuit."""
    qc = QuantumCircuit(2)
    m = taus.shape[0]
    
    # First local block
    a0 = angles[0]
    for q in range(2):
        qc.append(RXGate(a0[q, 0]), [q])
        qc.append(RYGate(a0[q, 1]), [q])
        qc.append(RZGate(a0[q, 2]), [q])

    # Entanglers and subsequent local blocks
    for k in range(m):
        qc.append(RZZGate(taus[k]), [0, 1])
        a = angles[k+1]
        for q in range(2):
            qc.append(RXGate(a[q, 0]), [q])
            qc.append(RYGate(a[q, 1]), [q])
            qc.append(RZGate(a[q, 2]), [q])
            
    return qc

# =========================================================
# === Qiskit Unitary Synthesis Plugin (The Wrapper)
# =========================================================
class PSFUnitarySynthesisPlugin(UnitarySynthesisPlugin):
    """
    A Qiskit UnitarySynthesisPlugin implementation for PSF-Zero v1.2.
    Numerically synthesizes 2Q unitaries into low-dissipation native circuits.
    """

    @property
    def max_qubits(self) -> int:
        return 2

    @property
    def min_qubits(self) -> int:
        return 2

    @property
    def supported_bases(self) -> list[list[str]]:
        return [['rx', 'ry', 'rz', 'rzz']]

    @property
    def supports_basis_exploration(self) -> bool:
        return False

    @property
    def supports_coupling_map(self) -> bool:
        return False

    @property
    def supports_natural_direction(self) -> bool:
        return False

    @property
    def supports_pulse_optimize(self) -> bool:
        return False

    @property
    def supports_target(self) -> bool:
        return False

    def run(self, unitary: np.ndarray, **options) -> QuantumCircuit:
        """
        Synthesize a unitary matrix into a QuantumCircuit.
        Pass custom hyperparameters via the options dictionary.
        """
        # Safely extract valid options for PSFHyper
        valid_keys = {f.name for f in dataclasses.fields(PSFHyper)}
        filtered_options = {k: v for k, v in options.items() if k in valid_keys}
        
        # Initialize hyperparams and synthesizer
        hyper = PSFHyper(**filtered_options)
        synth = PSFHybridSynthesizerFinal(hyper)
        
        # Run the optimization
        synth.run(unitary)
        
        if synth.angles is None or synth.taus is None:
            raise RuntimeError("Synthesis failed to find a valid parameter set.")
            
        # Convert the best found parameters into a Qiskit circuit
        return params_to_qiskit(synth.angles, synth.taus)
