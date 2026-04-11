from __future__ import annotations
import numpy as np
from dataclasses import dataclass, fields
from scipy.linalg import expm

from qiskit import QuantumCircuit
from qiskit.transpiler.passes.synthesis.plugin import UnitarySynthesisPlugin

# =========================================================
# Pauli matrices
# =========================================================
I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

# =========================================================
# Projective tools (Minimized damping)
# =========================================================
def projective_reg(x: np.ndarray) -> np.ndarray:
    return x / np.sqrt(1.0 + x**2)

def projective_grad(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + x**2)**1.5

# =========================================================
# Blocks
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
    m = len(taus)
    for l in range(m):
        U = local_block(angles[l]) @ U
        U = rzz_block(taus[l]) @ U
    U = local_block(angles[-1]) @ U
    return U

def F_avg(U: np.ndarray, V: np.ndarray) -> float:
    d = 4
    tr = np.trace(U.conj().T @ V)
    return float((np.abs(tr)**2 + d) / (d * (d + 1)))

# =========================================================
# Parameter-shift gradient
# =========================================================
def parameter_shift_grad(angles: np.ndarray, taus: np.ndarray, U_target: np.ndarray):
    g_a = np.zeros_like(angles)
    g_t = np.zeros_like(taus)
    shift = np.pi / 2.0
    m = len(taus)

    # Angles
    for l in range(m + 1):
        for q in range(2):
            for a in range(3):
                Ap = angles.copy()
                Am = angles.copy()
                Ap[l, q, a] += shift
                Am[l, q, a] -= shift
                g_a[l, q, a] = -0.5 * (
                    F_avg(compose_unitary(Ap, taus), U_target) -
                    F_avg(compose_unitary(Am, taus), U_target)
                )

    # Taus
    for l in range(m):
        tp = taus.copy()
        tm = taus.copy()
        tp[l] += shift
        tm[l] -= shift
        g_t[l] = -0.5 * (
            F_avg(compose_unitary(angles, tp), U_target) -
            F_avg(compose_unitary(angles, tm), U_target)
        )

    return g_a, g_t

# =========================================================
# KAK-style initialization
# =========================================================
def kak_init(m: int):
    angles = np.zeros((m + 1, 2, 3))
    taus = np.full(m, np.pi / 4 / m)  # Total entangling ≈ π/4
    return angles, taus

# =========================================================
# Hyperparameters (Optimal Balance)
# =========================================================
@dataclass
class PSFHyper:
    m: int = 5
    iters: int = 400
    lr_base: float = 0.15
    alpha_proj: float = 0.0005   # Minimal damping
    proj_every: int = 30
    seeds: int = 6

# =========================================================
# ULTIMATE OPTIMAL SYNTHESIZER
# =========================================================
class PSFHybridSynthesizerOptimal:
    def __init__(self, hyper: PSFHyper):
        self.hyper = hyper
        self.angles = None
        self.taus = None

    def run(self, U_target: np.ndarray) -> float:
        best_inf = float('inf')

        for seed in range(self.hyper.seeds):
            rng = np.random.default_rng(seed)
            angles, taus = kak_init(self.hyper.m)
            angles += rng.normal(scale=0.4, size=angles.shape)
            taus += rng.normal(scale=0.25, size=taus.shape)

            seed_best_inf = float('inf')
            seed_best_params = None

            for step in range(self.hyper.iters):
                # Cosine annealing learning rate
                lr = self.hyper.lr_base * 0.5 * (1 + np.cos(np.pi * step / self.hyper.iters))
                U = compose_unitary(angles, taus)
                inf = 1.0 - F_avg(U, U_target)

                if inf < seed_best_inf:
                    seed_best_inf = inf
                    seed_best_params = (angles.copy(), taus.copy())

                g_a, g_t = parameter_shift_grad(angles, taus, U_target)

                # Minimal projective damping
                g_a *= self.hyper.alpha_proj * projective_grad(angles)
                g_t *= self.hyper.alpha_proj * projective_grad(taus)

                angles -= lr * g_a
                taus -= lr * g_t

                # Thin out projections early on, enforce continuously in the end-game
                if step % self.hyper.proj_every == 0 or step > self.hyper.iters * 0.75:
                    angles = projective_reg(angles)
                    taus = projective_reg(taus)

            if seed_best_inf < best_inf:
                best_inf = seed_best_inf
                if seed_best_params is not None:
                    self.angles, self.taus = seed_best_params

        return best_inf

    def as_qiskit(self) -> QuantumCircuit:
        if self.angles is None or self.taus is None:
            raise ValueError("Run the synthesizer first.")
            
        qc = QuantumCircuit(2)
        m = len(self.taus)
        
        # Local → RZZ → Local ... → Last Local
        for l in range(m):
            a = self.angles[l]
            for q in range(2):
                qc.rx(a[q, 0], q)
                qc.ry(a[q, 1], q)
                qc.rz(a[q, 2], q)
            qc.rzz(self.taus[l], 0, 1)
            
        # Final local block
        a = self.angles[-1]
        for q in range(2):
            qc.rx(a[q, 0], q)
            qc.ry(a[q, 1], q)
            qc.rz(a[q, 2], q)
            
        return qc

# =========================================================
# === Qiskit Unitary Synthesis Plugin (The Wrapper)
# =========================================================
class PSFUnitarySynthesisPlugin(UnitarySynthesisPlugin):
    """
    A Qiskit UnitarySynthesisPlugin implementation for PSF-Zero Optimal v1.0.
    Numerically synthesizes 2Q unitaries with KAK initialization, Cosine Annealing,
    and end-game projective regularization for extreme fidelity.
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
        # Safely extract valid options for PSFHyper using dataclass fields
        valid_keys = {f.name for f in fields(PSFHyper)}
        filtered_options = {k: v for k, v in options.items() if k in valid_keys}
        
        hyper = PSFHyper(**filtered_options)
        synth = PSFHybridSynthesizerOptimal(hyper)
        
        synth.run(unitary)
        
        return synth.as_qiskit()
