#![allow(non_snake_case)]

use crate::algebra::{DenseFactorizationError, FactorSVD, FloatT, Matrix, ShapedMatrix};
use core::cmp::min;

#[derive(PartialEq, Eq)]
#[allow(dead_code)] //QRDecomposition is not used yet
pub(crate) enum SVDEngineAlgorithm {
    DivideAndConquer,
    QRDecomposition,
}

const DEFAULT_SVD_ALGORITHM: SVDEngineAlgorithm = SVDEngineAlgorithm::DivideAndConquer;

pub(crate) struct SVDEngine<T> {
    /// Computed singular values
    pub s: Vec<T>,

    /// Left and right SVD matrices, each containing.
    /// min(m,n) vectors.  Note right singular vectors
    /// are stored in transposed form.
    pub U: Matrix<T>,
    pub Vt: Matrix<T>,

    // BLAS factorization method
    pub algorithm: SVDEngineAlgorithm,

    // BLAS workspace (allocated vecs only)
    work: Vec<T>,
    iwork: Vec<i32>,
}

impl<T> SVDEngine<T>
where
    T: FloatT,
{
    pub fn new(size: (usize, usize)) -> Self {
        let (m, n) = size;
        let s = vec![T::zero(); min(m, n)];
        let U = Matrix::<T>::zeros((m, min(m, n)));
        let Vt = Matrix::<T>::zeros((min(m, n), n));
        let work = vec![T::one()];
        let iwork = vec![1];
        let algorithm = DEFAULT_SVD_ALGORITHM;
        Self {
            s,
            U,
            Vt,
            work,
            iwork,
            algorithm,
        }
    }
}

impl<T> FactorSVD for SVDEngine<T>
where
    T: FloatT,
{
    type T = T;
    fn svd(&mut self, A: &mut Matrix<Self::T>) -> Result<(), DenseFactorizationError> {
        let (m, n) = A.size();

        if self.U.nrows() != m || self.Vt.ncols() != n {
            return Err(DenseFactorizationError::IncompatibleDimension);
        }

        // standard BLAS ?gesdd and/or ?gesvd arguments for economy size SVD.

        let job = b'S'; // compact.
        let m = m.try_into().unwrap();
        let n = n.try_into().unwrap();
        let a = A.data_mut();
        let lda = m;
        let s = &mut self.s; // singular values go here
        let u = self.U.data_mut(); // U data goes here
        let ldu = m; // leading dim of U
        let vt = self.Vt.data_mut(); // Vt data goes here
        let ldvt = min(m, n); // leading dim of Vt
        let work = &mut self.work;
        let mut lwork = -1_i32; // -1 => config to request required work size
        let iwork = &mut self.iwork;
        let info = &mut 0_i32; // output info

        for i in 0..2 {
            // iwork is only used for the DivideAndConquer BLAS call
            // and should always be 8*min(m,n) elements in that case.
            // This will *not* shrink iwork in the case that the engines
            // algorithm is switched back and forth
            if self.algorithm == SVDEngineAlgorithm::DivideAndConquer {
                iwork.resize(8 * min(m, n) as usize, 0);
            }

            // Two calls to BLAS. First one gets size for work.
            match self.algorithm {
                SVDEngineAlgorithm::DivideAndConquer => T::xgesdd(
                    job, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, iwork, info,
                ),
                SVDEngineAlgorithm::QRDecomposition => T::xgesvd(
                    job, job, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, info,
                ),
            }
            if *info != 0 {
                return Err(DenseFactorizationError::SVD(*info));
            }

            // resize work vector and reset length
            if i == 0 {
                lwork = work[0].to_i32().unwrap();
                work.resize(lwork as usize, T::zero());
            }
        }
        Ok(())
    }
}

#[test]
fn test_svd() {
    use crate::algebra::{DenseMatrix, MultiplyGEMM, VectorMath};

    let mut A = Matrix::from(&[
        [3., 2., 2.],  //
        [2., 3., -2.], //
    ]);

    let Acopy = A.clone(); //A is corrupted after factorization

    let mut eng = SVDEngine::<f64>::new((2, 3));
    assert!(eng.svd(&mut A).is_ok());
    let sol = [5., 3.];
    assert!(eng.s.norm_inf_diff(&sol) < 1e-8);

    let mut M = Matrix::<f64>::zeros((2, 3));

    let U = &eng.U;
    let s = &eng.s;
    let Vt = &eng.Vt;

    //reconstruct matrix from SVD
    let mut Us = U.clone();
    for c in 0..Us.ncols() {
        for r in 0..Us.nrows() {
            Us[(r, c)] *= s[c];
        }
    }
    M.mul(&Us, Vt, 1.0, 0.0);

    assert!(M.data().norm_inf_diff(Acopy.data()) < 1e-8);
}
