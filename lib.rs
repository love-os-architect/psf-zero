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
