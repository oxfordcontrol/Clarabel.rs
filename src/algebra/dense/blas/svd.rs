#![allow(non_snake_case)]

use crate::algebra::*;
use core::cmp::min;
use std::iter::zip;

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

    pub fn resize(&mut self, size: (usize, usize)) {
        let (m, n) = size;
        self.s.resize(min(m, n), T::zero());
        self.U.resize((m, min(m, n)));
        self.Vt.resize((min(m, n), n));
    }
}

impl<T> FactorSVD<T> for SVDEngine<T>
where
    T: FloatT,
{
    fn factor<S>(&mut self, A: &mut DenseStorageMatrix<S, T>) -> Result<(), DenseFactorizationError>
    where
        S: AsMut<[T]> + AsRef<[T]>,
    {
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

    fn solve<S>(&mut self, B: &mut DenseStorageMatrix<S, T>)
    where
        S: AsMut<[T]> + AsRef<[T]>,
    {
        // get the dimensions for the SVD factors
        let m = self.U.nrows();
        let n = self.Vt.ncols();
        let k = min(m, n); //number of singular values

        // this function only implemented for square matrices
        // because otherwise writing the solution in place
        // does not make sense.   This is not a good general
        // implementation, but is only needed at present for a
        // rank-deficient, symmetric square solves in PSD
        // completion
        assert_eq!(m, n);

        // the number of columns in B
        let nrhs = B.ncols();
        assert_eq!(B.nrows(), m);

        // compute a tolerance for the singular values
        // to be considered invertible
        let tol = T::epsilon() * self.s[0].abs() * T::from(k).unwrap();

        // will compute B <- Vt * (Σ^-1 * (U^T * B))
        // we need a workspace that is at least nrhs * k
        // to hold the product C = U^T * B.  Will also
        // allocate additional space to hold the inverted
        // singular values
        let work = &mut self.work;
        work.resize(k + k * nrhs, T::zero());
        let (sinv, workC) = work.split_at_mut(k);

        // C <- U^T * B
        let mut C = BorrowedMatrixMut::from_slice_mut(workC, k, nrhs);
        C.mul(&self.U.t(), B, T::one(), T::zero());

        // C <- Σ^-1 * C
        zip(sinv.iter_mut(), self.s.iter()).for_each(|(sinv, s)| {
            if s.abs() > tol {
                *sinv = T::recip(s.abs());
            } else {
                *sinv = T::zero();
            }
        });

        for col in 0..nrhs {
            C.col_slice_mut(col).hadamard(sinv);
        }

        // B <- V * C
        B.mul(&self.Vt.t(), &C, T::one(), T::zero());
    }
}

macro_rules! generate_test_svd_factor {
    ($fxx:ty, $test_name:ident, $tolfn:ident) => {
        #[test]
        fn $test_name() {
            use crate::algebra::{DenseMatrix, MultiplyGEMM, VectorMath};

            let mut A = Matrix::<$fxx>::from(&[
                [3., 2., 2.],  //
                [2., 3., -2.], //
            ]);

            let Acopy = A.clone(); //A is corrupted after factorization

            let mut eng = SVDEngine::<$fxx>::new((2, 3));
            assert!(eng.factor(&mut A).is_ok());
            let sol = [5., 3.];
            assert!(eng.s.norm_inf_diff(&sol) < (1e-12 as $fxx).$tolfn());

            let mut M = Matrix::<$fxx>::zeros((2, 3));

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
            assert!(M.data().norm_inf_diff(Acopy.data()) < (1e-12 as $fxx).$tolfn());
        }
    };
}

generate_test_svd_factor!(f32, test_svd_factor_f32, sqrt);
generate_test_svd_factor!(f64, test_svd_factor_f64, abs);

macro_rules! generate_test_svd_solve {
    ($fxx:ty, $test_name:ident, $tolfn:ident) => {
        #[test]
        fn $test_name() {
            use crate::algebra::{DenseMatrix, VectorMath};

            // Singular and non-square A
            let mut A = Matrix::<$fxx>::from(&[
                [2., 4., 6.], //
                [1., 2., 3.], //
                [0., 1., 2.],
            ]);

            let mut B = Matrix::<$fxx>::from(&[
                [1., 2.], //
                [3., 4.],
                [5., 6.],
            ]);

            // this appears to be an exact solution
            let mut X = Matrix::<$fxx>::from(&[
                [-175., -200.], //
                [-40., -44.],
                [95., 112.],
            ]);
            X.data.scale(1. / 30.);

            let mut eng = SVDEngine::<$fxx>::new((3, 3));

            assert!(eng.factor(&mut A).is_ok());

            eng.solve(&mut B);
            assert!(B.data().norm_inf_diff(X.data()) < (1e-10 as $fxx).$tolfn());
        }
    };
}

generate_test_svd_solve!(f32, test_svd_solve_f32, sqrt);
generate_test_svd_solve!(f64, test_svd_solve_f64, abs);
