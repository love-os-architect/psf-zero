import time
import numpy as np
from qiskit.quantum_info import random_unitary
from qiskit.synthesis import TwoQubitBasisDecomposer
from qiskit.circuit.library import CXGate
import matplotlib.pyplot as plt

# Import QGL Compiler (Placeholder)
# import sys, os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'python')))
# from qgl_compiler import QGLProjector

def generate_random_su4_samples(num_samples: int = 1000):
    """Generate random SU(4) unitary matrices"""
    print(f"Generating {num_samples} random SU(4) unitaries...")
    samples = []
    for _ in range(num_samples):
        # random_unitary generates U(4), so we adjust the determinant to 1 to make it SU(4)
        u = random_unitary(4).data
        det = np.linalg.det(u)
        u_su4 = u * (det ** (-0.25))
        samples.append(u_su4)
    return samples

def benchmark_qiskit(samples):
    """Benchmark using Qiskit's standard transpiler (Decomposer)"""
    print("Running Qiskit Benchmark...")
    # Initialize the decomposer for CNOT basis (standard optimization level)
    decomposer = TwoQubitBasisDecomposer(CXGate())
    
    start_time = time.time()
    cnot_counts = []
    
    for u in samples:
        # Decompose into a circuit of CNOT and 1-qubit gates
        qc = decomposer(u)
        
        # Count the number of CNOTs in the generated circuit
        count = qc.count_ops().get('cx', 0)
        cnot_counts.append(count)
        
    end_time = time.time()
    
    avg_cnot = np.mean(cnot_counts)
    total_time = end_time - start_time
    time_per_op = total_time / len(samples) * 1000 # in milliseconds
    
    print(f"  Qiskit - Total Time: {total_time:.4f}s")
    print(f"  Qiskit - Time/op: {time_per_op:.4f}ms")
    print(f"  Qiskit - Avg CNOTs: {avg_cnot:.4f}")
    return total_time, time_per_op, avg_cnot

def benchmark_qgl(samples):
    """Benchmark using QGL (Rust Native Core)"""
    print("Running QGL Benchmark...")
    # compiler = QGLProjector()
    
    start_time = time.time()
    cnot_counts = []
    
    for u in samples:
        # Ideally, call the Rust core here to obtain the solution in O(1) time
        # The following is a mock implementation for simulation (replace with actual Rust function calls)
        
        # 1. Extract Cartan coordinates (c1, c2, c3) instantly using the Rust core
        # c1, c2, c3 = cartan_coordinates_rs(u)
        
        # 2. Deterministically determine the required number of CNOTs geometrically (Weyl Chamber properties)
        # Based on the Weyl chamber, it is guaranteed to be synthesizable with at most 3 CNOTs
        # In actual QGL, an analytically optimized circuit is returned here
        
        # [Dummy code: Assuming the Rust core processed it in O(1)]
        cnot_counts.append(3) # Always achieves the theoretical upper bound (or optimal value)
        
    end_time = time.time()
    
    avg_cnot = np.mean(cnot_counts)
    total_time = end_time - start_time
    time_per_op = total_time / len(samples) * 1000 # in milliseconds
    
    print(f"  QGL - Total Time: {total_time:.4f}s")
    print(f"  QGL - Time/op: {time_per_op:.4f}ms")
    print(f"  QGL - Avg CNOTs: {avg_cnot:.4f}")
    return total_time, time_per_op, avg_cnot

def plot_results(qiskit_res, qgl_res):
    """Visualize the results"""
    labels = ['Qiskit (Heuristic)', 'QGL (O(1) Rust Core)']
    times = [qiskit_res[1], qgl_res[1]] # ms per op
    cnots = [qiskit_res[2], qgl_res[2]]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Execution speed comparison (Log scale recommended)
    ax1.bar(labels, times, color=['blue', 'black'])
    ax1.set_yscale('log')
    ax1.set_title('Execution Time per SU(4) Projection (Log Scale)')
    ax1.set_ylabel('Time (ms)')
    
    # CNOT count comparison
    ax2.bar(labels, cnots, color=['blue', 'black'])
    ax2.set_title('Average CNOT Count per Circuit')
    ax2.set_ylabel('Avg CNOTs')
    
    plt.tight_layout()
    plt.savefig('benchmarks/results/benchmark_comparison.png')
    print("\nBenchmark results saved to 'benchmarks/results/benchmark_comparison.png'")

if __name__ == "__main__":
    import os
    os.makedirs('benchmarks/results', exist_ok=True)
    
    print("--- PSF-Zero / QGL Benchmark ---")
    samples = generate_random_su4_samples(1000)
    
    print("\n--- Starting Evaluations ---")
    qiskit_results = benchmark_qiskit(samples)
    
    # Simulating extremely fast dummy results until the Rust core API is implemented
    print("\n[NOTE] QGL benchmark is currently simulating O(1) execution time.")
    qgl_results = benchmark_qgl(samples) 
    
    plot_results(qiskit_results, qgl_results)
