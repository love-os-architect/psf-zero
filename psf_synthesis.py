from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional

from qiskit import QuantumCircuit
from qiskit.circuit.library import RZZGate, RXGate, RYGate, RZGate
from qiskit.transpiler.passes.synthesis.plugin import UnitarySynthesisPlugin

# -----------------------------------------
# === Core Math: /0 Projection & Unitaries
# -----------------------------------------

def projective_reg(vec: np.ndarray) -> float:
    """Projective regularization: x -> x/sqrt(1+x^2). Returns scalar penalty."""
    u = vec / np.sqrt(1.0 + vec**2)
    return float(np.sum(u**2))

def projective_grad(vec: np.ndarray) -> np.ndarray:
    """Analytic gradient (elementwise) of the projective regularizer."""
    return 2.0 * vec / (1.0 + vec**2)**2

def Rz(theta: float) -> np.ndarray:
    return np.array([[np.exp(-1j*theta/2), 0],
                     [0, np.exp(1j*theta/2)]], dtype=complex)

def Rx(theta: float) -> np.ndarray:
    c, s = np.cos(theta/2), -1j*np.sin(theta/2)
    return np.array([[c, s], [s, c]], dtype=complex)

def Ry(theta: float) -> np.ndarray:
    c, s = np.cos(theta/2), np.sin(theta/2)
    return np.array([[c, -s], [s, c]], dtype=complex)

def kron(*ops: np.ndarray) -> np.ndarray:
    M = np.array([[1]], dtype=complex)
    for op in ops:
        M = np.kron(M, op)
    return M

def Uzz(tau: float) -> np.ndarray:
    """Entangler: exp(-i tau/2 * Z⊗Z)"""
    ph = np.exp(-1j * 0.5 * tau)
    phc = np.conj(ph)
    return np.diag([ph, phc, phc, ph])

def local_block(block_params: np.ndarray) -> np.ndarray:
    (ax0, ay0, az0) = block_params[0]
    (ax1, ay1, az1) = block_params[1]
    U0 = Rz(az0) @ Ry(ay0) @ Rx(ax0)
    U1 = Rz(az1) @ Ry(ay1) @ Rx(ax1)
    return kron(U0, U1)

def compose_circuit_unitary(params_angles: np.ndarray, taus: np.ndarray) -> np.ndarray:
    U = local_block(params_angles[0])
    for k in range(len(taus)):
        U = Uzz(taus[k]) @ U
        U = local_block(params_angles[k+1]) @ U
    return U

def _polar_unitary(U: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Optional: project to the nearest unitary via symmetric polar decomposition.
    Keeps global phase behavior; use only if numerical drift is severe.
    """
    G = U.conj().T @ U
    w, V = np.linalg.eigh(G)
    inv_sqrt = (V @ np.diag(1.0/np.sqrt(np.clip(w, eps, None))) @ V.conj().T)
    return U @ inv_sqrt

def F_avg(U: np.ndarray, V: np.ndarray, polar_fix: bool = False) -> float:
    """Average gate fidelity between two unitaries in SU(4)."""
    if polar_fix:
        U = _polar_unitary(U)
        V = _polar_unitary(V)
    d = U.shape[0]
    tr = np.trace(U.conj().T @ V)
    return float((np.abs(tr)**2 + d) / (d*(d+1)))

PS = np.pi / 2.0  # Parameter-shift constant

# -----------------------------------------
# === Hyperparameters & Synthesizer Core
# -----------------------------------------

@dataclass
class PSFHyper:
    m: int = 3
    iters: int = 150
    lr: float = 0.2
    alpha_proj: float = 1e-2   # /0 regularization weight
    beta_H: float = 5e-3       # L1 penalty (dissipation proxy)
    beta_TV: float = 5e-3      # Total variation penalty (smooth pulses)
    seed: int = 42
    use_polar_fix: bool = False
    # Adam & step control
    betas: Tuple[float, float] = (0.9, 0.999)
    eps_adam: float = 1e-8
    step_clip: float = 1.5     # optional global step L2 cap (radians)

class PSFHybridSynthesizer:
    """
    Synthesizes a 2-qubit unitary into a low-dissipation native circuit:
      [Local] – Uzz – [Local] – ... – Uzz – [Local]
    with /0 projective regularization + analytic H/TV gradients + Adam.
    """
    def __init__(self, hyper: PSFHyper):
        self.h = hyper
        rng = np.random.default_rng(hyper.seed)
        self.params_angles = rng.normal(scale=0.2, size=(hyper.m+1, 2, 3))
        self.taus = rng.normal(scale=0.2, size=hyper.m)

        # Adam states on flat vector
        n = self._flat().size
        self.m1 = np.zeros(n, dtype=float)
        self.m2 = np.zeros(n, dtype=float)
        self.step_count = 0

    def _flat(self) -> np.ndarray:
        return np.concatenate([self.params_angles.ravel(), self.taus.ravel()])

    def _set_flat(self, vec: np.ndarray):
        nb = (self.h.m + 1) * 2 * 3
        self.params_angles = vec[:nb].reshape(self.h.m + 1, 2, 3)
        self.taus = vec[nb:]

    def _analytic_htv_grad(self) -> np.ndarray:
        # L1 part
        g_angles = np.sign(self.params_angles)
        g_taus = np.sign(self.taus)

        # TV for angles over block axis b
        diff_a = np.diff(self.params_angles, axis=0)
        sgn_a = np.sign(diff_a)

        tv_angles = np.zeros_like(self.params_angles)
        tv_angles[0] -= sgn_a[0]
        tv_angles[-1] += sgn_a[-1]
        if self.h.m > 1:
            tv_angles[1:-1] += sgn_a[:-1] - sgn_a[1:]

        # TV for taus along k
        if self.h.m >= 2:
            diff_t = np.diff(self.taus)
            sgn_t = np.sign(diff_t)
            tv_taus = np.zeros_like(self.taus)
            tv_taus[0]     -= sgn_t[0]
            tv_taus[-1]    += sgn_t[-1]
            if self.h.m > 2:
                tv_taus[1:-1] += sgn_t[:-1] - sgn_t[1:]
        else:
            tv_taus = np.zeros_like(self.taus)

        # combine with weights
        g_angles = self.h.beta_H * g_angles + self.h.beta_TV * tv_angles
        g_taus   = self.h.beta_H * g_taus   + self.h.beta_TV * tv_taus

        return np.concatenate([g_angles.ravel(), g_taus.ravel()])

    def _ps_grad(self, U_target: np.ndarray) -> np.ndarray:
        base = self._flat().copy()
        nb_angles = (self.h.m + 1) * 2 * 3
        grad = np.zeros_like(base)

        # Parameter-shift: angles
        for b in range(self.h.m + 1):
            for q in range(2):
                for c in range(3):
                    idx = b*2*3 + q*3 + c
                    v = base.copy(); v[idx] += PS; self._set_flat(v)
                    Lp = 1.0 - F_avg(compose_circuit_unitary(self.params_angles, self.taus),
                                     U_target, self.h.use_polar_fix)
                    v = base.copy(); v[idx] -= PS; self._set_flat(v)
                    Lm = 1.0 - F_avg(compose_circuit_unitary(self.params_angles, self.taus),
                                     U_target, self.h.use_polar_fix)
                    grad[idx] = 0.5 * (Lp - Lm)

        # Parameter-shift: entanglers
        for k in range(self.h.m):
            idx = nb_angles + k
            v = base.copy(); v[idx] += PS; self._set_flat(v)
            Lp = 1.0 - F_avg(compose_circuit_unitary(self.params_angles, self.taus),
                             U_target, self.h.use_polar_fix)
            v = base.copy(); v[idx] -= PS; self._set_flat(v)
            Lm = 1.0 - F_avg(compose_circuit_unitary(self.params_angles, self.taus),
                             U_target, self.h.use_polar_fix)
            grad[idx] = 0.5 * (Lp - Lm)

        # /0 projective grad (analytic)
        grad += self.h.alpha_proj * projective_grad(base)

        # H/TV analytic subgradient
        grad += self._analytic_htv_grad()

        self._set_flat(base)
        return grad

    def _adam_step(self, g: np.ndarray, lr: float):
        self.step_count += 1
        b1, b2 = self.h.betas
        self.m1 = b1 * self.m1 + (1 - b1) * g
        self.m2 = b2 * self.m2 + (1 - b2) * (g * g)
        m1_hat = self.m1 / (1 - b1**self.step_count)
        m2_hat = self.m2 / (1 - b2**self.step_count)
        step = lr * m1_hat / (np.sqrt(m2_hat) + self.h.eps_adam)

        if np.linalg.norm(step) > self.h.step_clip:
            step *= (self.h.step_clip / (np.linalg.norm(step) + 1e-12))
        return step

    def run(self, U_target: np.ndarray, tol: float = 1e-7):
        lr0 = self.h.lr
        last_loss = None
        for t in range(self.h.iters):
            lr = max(1e-4, lr0 * (0.5 * (1 + np.cos(np.pi * t / self.h.iters))))
            grad = self._ps_grad(U_target)
            step = self._adam_step(grad, lr)

            vec = self._flat()
            vec -= step
            self._set_flat(vec)

            if (t % 5) == 0:
                U = compose_circuit_unitary(self.params_angles, self.taus)
                loss = 1.0 - F_avg(U, U_target, self.h.use_polar_fix)
                if last_loss is not None and abs(last_loss - loss) < tol:
                    break
                last_loss = loss

    def as_qiskit(self) -> QuantumCircuit:
        qc = QuantumCircuit(2)
        a0 = self.params_angles[0]
        for q in range(2):
            qc.append(RXGate(a0[q][0]), [q])
            qc.append(RYGate(a0[q][1]), [q])
            qc.append(RZGate(a0[q][2]), [q])

        for k in range(self.h.m):
            qc.append(RZZGate(self.taus[k]), [0, 1])
            a = self.params_angles[k+1]
            for q in range(2):
                qc.append(RXGate(a[q][0]), [q])
                qc.append(RYGate(a[q][1]), [q])
                qc.append(RZGate(a[q][2]), [q])
        return qc

# -----------------------------------------
# === Qiskit Unitary Synthesis Plugin
# -----------------------------------------

class PSFUnitarySynthesisPlugin(UnitarySynthesisPlugin):
    """
    A UnitarySynthesisPlugin implementation for PSF-Zero.
    Numerically synthesizes 2Q unitaries into parameterized local blocks
    and RZZ entanglers using bounded projective regularization.
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
        """
        hyper = PSFHyper()
        synth = PSFHybridSynthesizer(hyper)
        
       
        synth.run(unitary)
        
        return synth.as_qiskit()
