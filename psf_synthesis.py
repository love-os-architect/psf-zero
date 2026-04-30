
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, fields

from qiskit import QuantumCircuit
from qiskit.transpiler.passes.synthesis.plugin import UnitarySynthesisPlugin

# =========================================================
# The Frictionless Core (Rust Native Integration)
# =========================================================
# NOTE: Heavy operations and absolute coordinate extraction 
# are entirely delegated to the deterministic Rust core.
from psf_zero_core import cartan_coordinates_full


# =========================================================
# Hyperparameters (Purified)
# =========================================================
@dataclass
class GeodesicPSFHyper:
    """
    Hyperparameters for the Z-axis geometric synthesizer.
    All variables related to stochastic search (iters, lr, m) 
    have been removed. Only convergence tolerances remain.
    """
    tol: float = 1e-9


# =========================================================
# FINAL SYNTHESIZER (Strictly Deterministic)
# =========================================================
class SU4GeodesicPSFSynthesizer:
    def __init__(self, hyper: GeodesicPSFHyper):
        self.hyper = hyper

    def synthesize(self, U_target: np.ndarray) -> QuantumCircuit:
        """
        Main synthesis flow.
        1. Rust core extracts absolute KAK decomposition (K1, A, K2)
        2. Convert to native single-shot RX/RY/RZ/RZZ circuit
        """
        # 1. Absolute KAK Extraction (Rust)
        # Returns the non-local core (c1, c2, c3) and the local 
        # SU(2)xSU(2) rotations K1 and K2 in Euler angles.
        k1_angles, cartan_core, k2_angles = cartan_coordinates_full(U_target)
        c1, c2, c3 = cartan_core

        # 2. Build native Qiskit circuit
        qc = QuantumCircuit(2)

        # ---------------------------------------------------
        # [Local Rotations K1] (Pre-Cartan)
        # ---------------------------------------------------
        qc.rz(k1_angles[0][0], 0)
        qc.ry(k1_angles[0][1], 0)
        qc.rz(k1_angles[0][2], 0)

        qc.rz(k1_angles[1][0], 1)
        qc.ry(k1_angles[1][1], 1)
        qc.rz(k1_angles[1][2], 1)

        # ---------------------------------------------------
        # [Non-Local Cartan Core A(c1, c2, c3)]
        # Applied in a single shot. No loops. No amplified entanglement.
        # ---------------------------------------------------
        qc.rxx(2 * c1, 0, 1)
        qc.ryy(2 * c2, 0, 1)
        qc.rzz(2 * c3, 0, 1)

        # ---------------------------------------------------
        # [Local Rotations K2] (Post-Cartan)
        # ---------------------------------------------------
        qc.rz(k2_angles[0][0], 0)
        qc.ry(k2_angles[0][1], 0)
        qc.rz(k2_angles[0][2], 0)

        qc.rz(k2_angles[1][0], 1)
        qc.ry(k2_angles[1][1], 1)
        qc.rz(k2_angles[1][2], 1)

        return qc


# =========================================================
# Qiskit Official Plugin
# =========================================================
class SU4GeodesicPSFUnitarySynthesis(UnitarySynthesisPlugin):
    """
    Official Qiskit plugin.
    Can be registered and used transparently in PassManager.
    """
    @property
    def max_qubits(self) -> int:
        return 2

    @property
    def min_qubits(self) -> int:
        return 2

    @property
    def supported_bases(self) -> list[str]:
        # Native bases required to express the full KAK decomposition
        return ['rx', 'ry', 'rz', 'rxx', 'ryy', 'rzz']

    def run(self, unitary: np.ndarray, **options) -> QuantumCircuit:
        valid = {f.name for f in fields(GeodesicPSFHyper)}
        hyper_kwargs = {k: v for k, v in options.items() if k in valid}
        hyper = GeodesicPSFHyper(**hyper_kwargs)

        synth = SU4GeodesicPSFSynthesizer(hyper)
        return synth.synthesize(unitary)
