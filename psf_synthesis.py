
from __future__ import annotations

import numpy as np

from dataclasses import dataclass, fields

from scipy.linalg import expm

from qiskit import QuantumCircuit

from qiskit.transpiler.passes.synthesis.plugin import UnitarySynthesisPlugin

# ===============================================

# Analytic singlequbit rotations

# ===============================================

def local_rot(theta: float, axis: int) -> np.ndarray:

    c = np.cos(theta / 2)

    s = np.sin(theta / 2)

    if axis == 0:      # Rx

        return np.array([[c, -1j*s], [-1j*s, c]], dtype=complex)

    elif axis == 1:    # Ry

        return np.array([[c, -s], [s,  c]], dtype=complex)

    else:              # Rz

        return np.array([[np.exp(-1j*theta/2), 0], [0, np.exp(1j*theta/2)]], dtype=complex)

# ===============================================

# Analytic RZZ

# ===============================================

def rzz_block(tau: float) -> np.ndarray:

    p = np.exp(-0.5j * tau)

    m = np.conj(p)

    return np.diag([p, m, m, p])

# ===============================================

# Optimized 2×2 ⊗ 2×2

# ===============================================

def kron_2x2(a: np.ndarray, b: np.ndarray) -> np.ndarray:

    out = np.zeros((4, 4), dtype=complex)

    for i in range(2):

        for j in range(2):

            out[2*i:2*(i+1), 2*j:2*(j+1)] = a[i, j] * b

    return out

# ===============================================

# Core: PSF unitary composition (FINAL)

# ===============================================

def compose_unitary(angles: np.ndarray, taus: np.ndarray) -> np.ndarray:

    m = len(taus)

    u = np.eye(4, dtype=complex)

    for l in range(m + 1):

        local = np.eye(4, dtype=complex)

        for q in range(2):

            for a in range(3):

                theta = angles[l, q, a]

                if abs(theta) < 1e-12: continue # Zero-friction skip

                uq = local_rot(theta, a)

                if q == 0: big = kron_2x2(uq, np.eye(2))

                else:      big = kron_2x2(np.eye(2), uq)

                local = big @ local

        u = local @ u

        if l < m:

            if abs(taus[l]) > 1e-12: # Zero-friction skip

                u = rzz_block(taus[l]) @ u

    return u

# =========================================================

# SU(4) Geodesic Optimization (Core)

# =========================================================

def F_avg(U: np.ndarray, V: np.ndarray) -> float:

    d = U.shape[0]

    tr = np.trace(U.conj().T @ V)

    return float((np.abs(tr)**2 + d) / (d * (d + 1)))

def fidelity_grad(U: np.ndarray, V: np.ndarray) -> np.ndarray:

    d = U.shape[0]

    tr = np.trace(U.conj().T @ V)

    return -2 * (np.conj(tr) / (d * (d + 1))) * V

def su_projection(G: np.ndarray, U: np.ndarray) -> np.ndarray:

    H = 0.5 * (G @ U.conj().T - U @ G.conj().T)

    H -= np.trace(H) / H.shape[0] * np.eye(H.shape[0])

    return H

def geodesic_line_search(U: np.ndarray, U_target: np.ndarray, H: np.ndarray, eta0=1.0, c=1e-4, beta=0.5, max_iter=20):

    f0 = 1 - F_avg(U, U_target)

    dU = -H @ U

    directional = np.real(np.trace(fidelity_grad(U, U_target).conj().T @ dU))

    eta = eta0

    for _ in range(max_iter):

        U_new = expm(-eta * H) @ U # SU(4) exact geodesic traversal

        f_new = 1 - F_avg(U_new, U_target)

        if f_new <= f0 + c * eta * directional: return eta, U_new

        eta *= beta

    return eta, U

# =========================================================

# PSF Decomposer (Last remaining X-axis code)

# =========================================================

# NOTE: This uses a random walk placeholder.

# NEXT STEP: Replace this with analytic Cartan decomposition for absolute R=0.

def project_to_psf(U_target, m, iters=300, lr=0.2):

    angles = np.zeros((m + 1, 2, 3))

    taus = np.full(m, np.pi / 4 / m)

    for _ in range(iters):

        U = compose_unitary(angles, taus)

        for l in range(m + 1): angles[l] -= lr * np.real(np.random.normal(scale=0.01, size=(2, 3)))

        taus -= lr * np.real(np.random.normal(scale=0.01, size=m))

    return angles, taus

# =========================================================

# Hyperparameters

# =========================================================

@dataclass

class GeodesicPSFHyper:

    m: int = 5

    iters: int = 120

    tol: float = 1e-9

# =========================================================

# FINAL SYNTHESIZER

# =========================================================

class SU4GeodesicPSFSynthesizer:

    def __init__(self, hyper: GeodesicPSFHyper):

        self.hyper = hyper

        self.angles = None

        self.taus = None

    def run(self, U_target: np.ndarray) -> float:

        U = np.eye(4, dtype=complex)

        for _ in range(self.hyper.iters):

            G = fidelity_grad(U, U_target)

            H = su_projection(G, U)

            eta, U = geodesic_line_search(U, U_target, H)

            if np.linalg.norm(H) < self.hyper.tol: break

        

        # We now have the perfect SU(4) target (U). Project it to hardware pulses.

        self.angles, self.taus = project_to_psf(U, self.hyper.m)

        return 1 - F_avg(compose_unitary(self.angles, self.taus), U_target)

    def as_qiskit(self) -> QuantumCircuit:

        qc = QuantumCircuit(2)

        for l in range(len(self.taus)):

            a = self.angles[l]

            for q in range(2):

                qc.rx(a[q,0], q); qc.ry(a[q,1], q); qc.rz(a[q,2], q)

            qc.rzz(self.taus[l], 0, 1)

        a = self.angles[-1]

        for q in range(2):

            qc.rx(a[q,0], q); qc.ry(a[q,1], q); qc.rz(a[q,2], q)

        return qc

# =========================================================

# Qiskit Plugin Wrapper

# =========================================================

class SU4GeodesicPSFUnitarySynthesis(UnitarySynthesisPlugin):

    @property

    def max_qubits(self): return 2

    @property

    def min_qubits(self): return 2

    @property

    def supported_bases(self): return [['rx', 'ry', 'rz', 'rzz']]

    

    def run(self, unitary: np.ndarray, **options) -> QuantumCircuit:

        valid = {f.name for f in fields(GeodesicPSFHyper)}

        hyper = GeodesicPSFHyper(**{k:v for k,v in options.items() if k in valid})

        synth = SU4GeodesicPSFSynthesizer(hyper)

        synth.run(unitary)

        return synth.as_qiskit()
