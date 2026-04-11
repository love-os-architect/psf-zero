from __future__ import annotations
import numpy as np
import dataclasses
from dataclasses import dataclass
from typing import Tuple, Optional
from scipy.linalg import expm, logm

from qiskit import QuantumCircuit
from qiskit.circuit.library import RZZGate, RXGate, RYGate, RZGate
from qiskit.transpiler.passes.synthesis.plugin import UnitarySynthesisPlugin

# =========================================================
# === Core Math: /0 Projection, SU(4) Geometry & Unitaries
# =========================================================
I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

def projective_reg(x: np.ndarray) -> np.ndarray:
    """Projective regularization: bounds parameters without hard clipping."""
    return x / np.sqrt(1.0 + x**2)

def projective_grad(x: np.ndarray) -> np.ndarray:
    """Analytic gradient (damping factor) of the projective regularizer."""
    return 1.0 / (1.0 + x**2) ** 1.5

def local_block(angles: np.ndarray) -> np.ndarray:
    """Local single-qubit rotations: RX-RY-RZ on both qubits."""
    U = np.eye(4, dtype=complex)
    paulis = [X, Y, Z]
    for q in range(2):
        for a in range(3):
            theta = angles[q, a]
            Uq = expm(-1j * theta / 2 * paulis[a])
            U = np.kron(Uq, I) @ U if q == 0 else np.kron(I, Uq) @ U
    return U

def rzz_block(tau: float) -> np.ndarray:
    """Entangler: exp(-i tau/2 * Z⊗Z)"""
    return expm(-1j * tau / 2 * np.kron(Z, Z))

def compose_unitary(angles: np.ndarray, taus: np.ndarray) -> np.ndarray:
    """Compose the full 2Q circuit unitary."""
    U = np.eye(4, dtype=complex)
    m = taus.shape[0]
    for l in range(m):
        U = local_block(angles[l]) @ U
        U = rzz_block(taus[l]) @ U
    U = local_block(angles[-1]) @ U
    return U

def F_avg(U: np.ndarray, V: np.ndarray) -> float:
    """Average gate fidelity between two unitaries in SU(4)."""
    d = U.shape[0]
    return float((np.abs(np.trace(U.conj().T @ V))**2 + d) / (d * (d + 1)))

def geodesic_loss(U: np.ndarray, U_target: np.ndarray) -> float:
    """SU(4) geodesic deviation loss to avoid barren plateaus."""
    delta = logm(U_target.conj().T @ U)
    return float(np.real(np.trace(delta.conj().T @ delta)))

def hardware_tau_penalty(taus: np.ndarray, tau_cal: np.ndarray) -> float:
    """Hardware prior: penalize deviation from calibrated RZZ duration."""
    return float(np.sum((taus - tau_cal) ** 2))

# =========================================================
# === Parameter-Shift Gradients
# =========================================================
def parameter_shift_grad(angles: np.ndarray, taus: np.ndarray, U_target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    m = taus.shape[0]
    g_angles = np.zeros_like(angles)
    g_taus   = np.zeros_like(taus)
    shift = np.pi / 2.0

    # Angles (Local blocks)
    for l in range(m + 1):
        for q in range(2):
            for a in range(3):
                Ap, Am = angles.copy(), angles.copy()
                Ap[l, q, a] += shift
                Am[l, q, a] -= shift
                Fp = F_avg(compose_unitary(Ap, taus), U_target)
                Fm = F_avg(compose_unitary(Am, taus), U_target)
                g_angles[l, q, a] = -0.5 * (Fp - Fm)

    # Taus (RZZ blocks)
    for l in range(m):
        tp, tm = taus.copy(), taus.copy()
        tp[l] += shift
        tm[l] -= shift
        Fp = F_avg(compose_unitary(angles, tp), U_target)
        Fm = F_avg(compose_unitary(angles, tm), U_target)
        g_taus[l] = -0.5 * (Fp - Fm)

    return g_angles, g_taus

# =========================================================
# === Hyperparameters & Synthesizer Core
# =========================================================
@dataclass
class PSFHyper:
    m: int = 3
    iters: int = 150
    lr: float = 0.15
    beta_L1: float = 1e-3      # Dissipation proxy
    beta_TV: float = 1e-3      # Smooth pulses
    beta_geo: float = 1e-2     # Geodesic steering
    beta_hw: float = 1e-2      # Hardware RZZ calibration steering
    alpha_proj: float = 5e-3   # /0 regularization
    seeds: int = 3             # Multi-seed for global minimum
    tau_cal: Optional[np.ndarray] = None # e.g. np.array([np.pi/4]*m)

class PSFHybridSynthesizerFinal:
    """
    Synthesizes a 2-qubit unitary using Geodesic Loss, Hardware Priors, 
    and Projective Regularization with Multi-seed SGD.
    """
    def __init__(self, hyper: PSFHyper):
        self.hyper = hyper
        self.best_angles = None
        self.best_taus = None

    def run(self, U_target: np.ndarray) -> float:
        best_loss_overall = np.inf

        # Multi-seed global search
        for seed in range(self.hyper.seeds):
            rng = np.random.default_rng(seed)
            angles = rng.normal(scale=0.1, size=(self.hyper.m + 1, 2, 3))
            taus   = rng.normal(scale=0.1, size=(self.hyper.m,))

            for _ in range(self.hyper.iters):
                U = compose_unitary(angles, taus)

                # Forward Loss Calculation
                loss = 1.0 - F_avg(U, U_target)
                loss += self.hyper.beta_geo * geodesic_loss(U, U_target)
                loss += self.hyper.beta_L1 * np.sum(np.abs(angles))
                loss += self.hyper.beta_TV * np.sum(np.abs(np.diff(angles, axis=0)))
                
                if self.hyper.tau_cal is not None:
                    loss += self.hyper.beta_hw * hardware_tau_penalty(taus, self.hyper.tau_cal)

                # Gradient Calculation & Penalties
                g_a, g_t = parameter_shift_grad(angles, taus, U_target)

                g_a += self.hyper.beta_L1 * np.sign(angles)
                g_a[:-1] += self.hyper.beta_TV * np.sign(angles[:-1] - angles[1:])
                g_a[1:]  -= self.hyper.beta_TV * np.sign(angles[:-1] - angles[1:])

                if self.hyper.tau_cal is not None:
                    g_t += 2.0 * self.hyper.beta_hw * (taus - self.hyper.tau_cal)

                # Projective Damping
                g_a *= projective_grad(angles)
                g_t *= projective_grad(taus)

                # SGD Step
                angles -= self.hyper.lr * g_a
                taus   -= self.hyper.lr * g_t

                # Projective Bounds
                angles = projective_reg(angles)
                taus   = projective_reg(taus)

            # Store the best seed result
            if loss < best_loss_overall:
                best_loss_overall = loss
                self.best_angles = angles.copy()
                self.best_taus = taus.copy()

        return best_loss_overall

    def as_qiskit(self) -> QuantumCircuit:
        if self.best_angles is None or self.best_taus is None:
            raise ValueError("Synthesizer has not been run yet.")
            
        qc = QuantumCircuit(2)
        # First local block
        a0 = self.best_angles[0]
        for q in range(2):
            qc.append(RXGate(a0[q, 0]), [q])
            qc.append(RYGate(a0[q, 1]), [q])
            qc.append(RZGate(a0[q, 2]), [q])

        # Entanglers and subsequent local blocks
        for k in range(self.hyper.m):
            qc.append(RZZGate(self.best_taus[k]), [0, 1])
            a = self.best_angles[k+1]
            for q in range(2):
                qc.append(RXGate(a[q, 0]), [q])
                qc.append(RYGate(a[q, 1]), [q])
                qc.append(RZGate(a[q, 2]), [q])
        return qc

# =========================================================
# === Qiskit Unitary Synthesis Plugin
# =========================================================
class PSFUnitarySynthesisPlugin(UnitarySynthesisPlugin):
    """
    A Qiskit UnitarySynthesisPlugin implementation for PSF-Zero Final.
    Numerically synthesizes 2Q unitaries into low-dissipation native circuits 
    using geodesic targeting and hardware priors.
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
        
        # If tau_cal is provided as a list/tuple in options, convert to numpy
        if 'tau_cal' in filtered_options and filtered_options['tau_cal'] is not None:
            filtered_options['tau_cal'] = np.array(filtered_options['tau_cal'])

        hyper = PSFHyper(**filtered_options)
        synth = PSFHybridSynthesizerFinal(hyper)
        
        # Execute the optimization
        synth.run(unitary)
        
        # Return the resulting Qiskit circuit
        return synth.as_qiskit()
