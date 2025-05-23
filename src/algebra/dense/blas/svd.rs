#![allow(non_snake_case)]

use crate::algebra::*;
use core::cmp::min;
use std::iter::zip;

#[allow(dead_code)]
#[derive(PartialEq, Eq, Copy, Clone, Default)]
pub(crate) enum SVDEngineAlgorithm {
    #[default]
    DivideAndConquer,
    QRDecomposition,
}

pub(crate) struct SVDBlasWorkVectors<T> {
    pub work: Vec<T>,
    pub iwork: Vec<i32>,
}

impl<T: FloatT> Default for SVDBlasWorkVectors<T> {
    fn default() -> Self {
        // must be at least 1 element because the
        // requiring work size is written into the
        // first element
        let work = vec![T::one()];
        let iwork = vec![1];
        Self { work, iwork }
    }
}

pub(crate) struct SVDEngine<T> {
    /// Computed singular values
    pub s: Vec<T>,

    /// Left and right SVD matrices, each containing.
    /// min(m,n) vectors.  Note right singular vectors
    /// are stored in transposed form.
    pub U: Matrix<T>,
    pub Vt: Matrix<T>,

    // BLAS workspace (allocated vecs only)
    pub blas: Option<SVDBlasWorkVectors<T>>,

    // BLAS factorization method
    pub algorithm: SVDEngineAlgorithm,
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
        let blas = None;
        let algorithm = SVDEngineAlgorithm::default();
        Self {
            s,
            U,
            Vt,
            blas,
            algorithm,
        }
    }

    pub fn resize(&mut self, size: (usize, usize)) {
        let (m, n) = size;
        self.s.resize(min(m, n), T::zero());
        self.U.resize((m, min(m, n)));
        self.Vt.resize((min(m, n), n));
    }

    fn checkdim_factor<S>(
        &mut self,
        A: &mut DenseStorageMatrix<S, T>,
    ) -> Result<(), DenseFactorizationError>
    where
        S: AsMut<[T]> + AsRef<[T]>,
    {
        let (m, n) = A.size();

        if self.U.nrows() != m || self.Vt.ncols() != n {
            Err(DenseFactorizationError::IncompatibleDimension)
        } else {
            Ok(())
        }
    }

    fn checkdim_solve<S>(
        &mut self,
        B: &mut DenseStorageMatrix<S, T>,
    ) -> Result<(), DenseFactorizationError>
    where
        S: AsMut<[T]> + AsRef<[T]>,
    {
        // get the dimensions for the SVD factors
        let m = self.U.nrows();
        let n = self.Vt.ncols();

        // this function only implemented for square matrices
        // because otherwise writing the solution in place
        // does not make sense.   This is not a good general
        // implementation, but is only needed at present for a
        // rank-deficient, symmetric square solves in PSD
        // completion
        if m != n {
            return Err(DenseFactorizationError::IncompatibleDimension);
        }

        // the number of columns in B
        if B.nrows() != m {
            return Err(DenseFactorizationError::IncompatibleDimension);
        }
        Ok(())
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
        self.checkdim_factor(A)?;

        // all special cases are square
        if A.is_square() {
            match A.nrows() {
                1 => self.factor1(A),
                2 => self.factor2(A),
                3 => self.factor3(A),
                _ => self.factorblas(A),
            }
        } else {
            // non-square matrices
            self.factorblas(A)
        }
    }

    fn solve<S>(&mut self, B: &mut DenseStorageMatrix<S, T>)
    where
        S: AsMut<[T]> + AsRef<[T]>,
    {
        // just unwrap here.   The only way
        // to fail is to have a non-square matrix,
        // which is not of interest in the crate
        // and should panic if encountered.
        self.checkdim_solve(B).unwrap();

        // PJG: always use blas to solve, regardless of
        // dimension.  SVD solve does not happen over cones,
        // and is only I used (I think) during chordal
        // decomposition.   Could come back to this for
        // custom implementation if there is a bottleneck.
        // NB: note this means we always carry the blas
        // workspace, even at low dimensions.

        self.solveblas(B);
    }
}

// trivial implementation for 1x1 matrices
impl<T> SVDEngine<T>
where
    T: FloatT,
{
    fn factor1<S>(
        &mut self,
        A: &mut DenseStorageMatrix<S, T>,
    ) -> Result<(), DenseFactorizationError>
    where
        S: AsMut<[T]> + AsRef<[T]>,
    {
        self.U[(0, 0)] = T::one();
        self.Vt[(0, 0)] = T::one();
        self.s[0] = A[(0, 0)];

        if self.s[0] < T::zero() {
            self.s[0] = -self.s[0];
            self.U[(0, 0)] = -T::one();
        };
        Ok(())
    }
}

// implementation for 2x2 matrices

impl<T> SVDEngine<T>
where
    T: FloatT,
{
    fn factor2<S>(
        &mut self,
        A: &mut DenseStorageMatrix<S, T>,
    ) -> Result<(), DenseFactorizationError>
    where
        S: AsMut<[T]> + AsRef<[T]>,
    {
        let mut As = DenseMatrix2::<T>::from(A);
        let mut Vs = DenseMatrix2::<T>::zeros();
        let mut Us = DenseMatrix2::<T>::zeros();

        let s = As.svd(&mut Us, &mut Vs);
        self.s.copy_from_slice(&s);
        self.U.data.copy_from(&Us.data);

        // Vt is stored in transposed form
        Vs.transpose_in_place();
        self.Vt.copy_from_slice(&Vs.data);
        Ok(())
    }
}

// implementation for 3x3 matrices

impl<T> SVDEngine<T>
where
    T: FloatT,
{
    fn factor3<S>(
        &mut self,
        A: &mut DenseStorageMatrix<S, T>,
    ) -> Result<(), DenseFactorizationError>
    where
        S: AsMut<[T]> + AsRef<[T]>,
    {
        let mut As = DenseMatrix3::<T>::from(A);
        let mut Vs = DenseMatrix3::<T>::zeros();
        let mut Us = DenseMatrix3::<T>::zeros();

        let s = As.svd(&mut Us, &mut Vs);
        self.s.copy_from_slice(&s);
        self.U.data.copy_from(&Us.data);

        // Vt is stored in transposed form
        Vs.transpose_in_place();
        self.Vt.copy_from_slice(&Vs.data);
        Ok(())
    }
}

// implementation for arbitrary size (square) matrices

impl<T> SVDEngine<T>
where
    T: FloatT,
{
    fn factorblas<S>(
        &mut self,
        A: &mut DenseStorageMatrix<S, T>,
    ) -> Result<(), DenseFactorizationError>
    where
        S: AsMut<[T]> + AsRef<[T]>,
    {
        // standard BLAS ?gesdd and/or ?gesvd arguments for economy size SVD.

        let m = self.U.nrows();
        let n = self.Vt.ncols();

        // unwrap or populate on the first call
        let blaswork = self.blas.get_or_insert_with(SVDBlasWorkVectors::default);

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
        let work = &mut blaswork.work;
        let mut lwork = -1_i32; // -1 => config to request required work size
        let iwork = &mut blaswork.iwork;
        let info = &mut 0_i32; // output info

        for i in 0..2 {
            // iwork is only used for the DivideAndConquer BLAS call
            // and should always be 8*min(m,n) elements in that case.
            // This will *not* shrink iwork in the case that the engine's
            // algorithm is switched back and forth
            if self.algorithm == SVDEngineAlgorithm::DivideAndConquer {
                iwork.resize(8 * min(m, n) as usize, 0);
            }

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

    fn solveblas<S>(&mut self, B: &mut DenseStorageMatrix<S, T>)
    where
        S: AsMut<[T]> + AsRef<[T]>,
    {
        // get the dimensions for the SVD factors
        let m = self.U.nrows();
        let n = self.Vt.ncols();
        let k = min(m, n); //number of singular values

        // the number of columns in B
        let nrhs = B.ncols();

        // compute a tolerance for the singular values
        // to be considered invertible
        let tol = T::epsilon() * self.s[0].abs() * T::from(k).unwrap();

        // unwrap or populate on the first call
        let blaswork = self.blas.get_or_insert_with(SVDBlasWorkVectors::default);

        // will compute B <- Vt * (Σ^-1 * (U^T * B))
        // we need a workspace that is at least nrhs * k
        // to hold the product C = U^T * B.  Will also
        // allocate additional space to hold the inverted
        // singular values
        blaswork.work.resize(k + k * nrhs, T::zero());
        let (sinv, workC) = blaswork.work.split_at_mut(k);

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

// ---- unit testing ----

#[cfg(test)]
mod test {
    use super::*;

    fn test_solve_data_2x2<T: FloatT>() -> (Matrix<T>, Matrix<T>, Matrix<T>) {
        // Create a symmetric matrix S
        let A = Matrix::<T>::from(&[
            [(4.0).as_T(), (1.0).as_T()],
            [(1.0).as_T(), (3.0).as_T()],
        ]);
    
        // Solution matrix X with 2 columns
        let X = Matrix::<T>::from(&[
            [(2.0).as_T(), (3.0).as_T()],
            [(1.0).as_T(), (2.0).as_T()],
        ]);
    
        // Right-hand side B = S*X
        let B = Matrix::<T>::from(&[
            [(9.0).as_T(), (14.0).as_T()],
            [(5.0).as_T(), (9.0).as_T()],
        ]);
    
        (A, X, B)
    }
 
    #[rustfmt::skip]
    fn test_solve_data_3x3<T: FloatT>() -> (Matrix<T>, Matrix<T>, Matrix<T>) {
        let A = Matrix::<T>::from(&[
            [(8.0).as_T(), (-2.0).as_T(), (4.0).as_T()], 
            [(-2.0).as_T(), (12.0).as_T(), (2.0).as_T()], 
            [(4.0).as_T(), (2.0).as_T(), (6.0).as_T()]
        ]);

        let X = Matrix::<T>::from(&[
            [(1.0).as_T(), (2.0).as_T()], //
            [(3.0).as_T(), (4.0).as_T()], //
            [(5.0).as_T(), (6.0).as_T()],
        ]);

        let B = Matrix::<T>::from(&[
            [(22.0).as_T(), (32.0).as_T()], //
            [(44.0).as_T(), (56.0).as_T()], //
            [(40.0).as_T(), (52.0).as_T()],
        ]);

        (A, X, B)
    }

       
    #[rustfmt::skip]
    fn test_solve_data_4x4<T: FloatT>() -> (Matrix<T>, Matrix<T>, Matrix<T>) {
        // Create a symmetric matrix S
        let A = Matrix::<T>::from(&[
            [(10.0).as_T(), (2.0).as_T(),  (3.0).as_T(),  (1.0).as_T()],
            [(2.0).as_T(),  (8.0).as_T(),  (0.0).as_T(),  (3.0).as_T()],
            [(3.0).as_T(),  (0.0).as_T(),  (6.0).as_T(),  (2.0).as_T()],
            [(1.0).as_T(),  (3.0).as_T(),  (2.0).as_T(),  (9.0).as_T()],
        ]);

        // Solution matrix X with 2 columns
        let X = Matrix::<T>::from(&[
            [(1.0).as_T(), (2.0).as_T()],
            [(2.0).as_T(), (3.0).as_T()],
            [(3.0).as_T(), (1.0).as_T()],
            [(4.0).as_T(), (2.0).as_T()],
        ]);

        // Right-hand side B = S*X
        let B = Matrix::<T>::from(&[
            [(27.0).as_T(), (31.0).as_T()],
            [(30.0).as_T(), (34.0).as_T()],
            [(29.0).as_T(), (16.0).as_T()],
            [(49.0).as_T(), (31.0).as_T()],
        ]);

        (A, X, B)
    }

    fn run_svd_solve_test<T>(A: &Matrix<T>, X: &Matrix<T>, B: &Matrix<T>, tolfn: fn(T) -> T)
    where
        T: FloatT,
    {
        use crate::algebra::VectorMath;

        let methods = [
            SVDEngineAlgorithm::DivideAndConquer,
            SVDEngineAlgorithm::QRDecomposition,
        ];

        for method in methods.iter() {

            // A and B are modified inplace during factor/solve
            let mut thisA = A.clone();
            let mut thisB = B.clone();

            let mut eng = SVDEngine::<T>::new(thisA.size());
            eng.algorithm = *method;

            assert!(eng.factor(&mut thisA).is_ok());
            eng.solve(&mut thisB);

            assert!(thisB.data().norm_inf_diff(X.data()) < tolfn(1e-10.as_T()));
        }
    }

    macro_rules! generate_test_svd_solve {
        ($fxx:ty, $test_name:ident, $tolfn:ident) => {
            #[test]
            fn $test_name() {
                let (mut A, mut X, mut B) = test_solve_data_2x2::<$fxx>();
                run_svd_solve_test(&mut A, &mut X, &mut B,  |x| x.$tolfn());

                let (mut A, mut X, mut B) = test_solve_data_3x3::<$fxx>();  
                run_svd_solve_test(&mut A, &mut X, &mut B,  |x| x.$tolfn());

                let (mut A, mut X, mut B) = test_solve_data_4x4::<$fxx>();
                run_svd_solve_test(&mut A, &mut X, &mut B,  |x| x.$tolfn());
            }
        };
    }

    generate_test_svd_solve!(f32, test_svd_solve_f32, sqrt);
    generate_test_svd_solve!(f64, test_svd_solve_f64, abs);


    fn test_factor_data_2x2<T: FloatT>() ->Matrix<T> {
        let (A,_,_) = test_solve_data_2x2::<T>();
        A
    }
    fn test_factor_data_3x3<T: FloatT>() ->Matrix<T> {
        let (A,_,_) = test_solve_data_3x3::<T>();
        A
    }
    fn test_factor_data_4x4<T: FloatT>() ->Matrix<T> {
        let (A,_,_) = test_solve_data_4x4::<T>();
        A
    }

    #[rustfmt::skip]
    fn test_factor_data_2x4<T: FloatT>() -> Matrix<T> {
        Matrix::<T>::from(&[
            [(10.0).as_T(), (2.0).as_T(),  (3.0).as_T(),  (1.0).as_T()],
            [(2.0).as_T(),  (8.0).as_T(),  (0.0).as_T(),  (3.0).as_T()],
        ])
    }

    #[rustfmt::skip]
    fn test_factor_data_4x2<T: FloatT>() -> Matrix<T> {
        Matrix::<T>::from(&[
            [(10.0).as_T(), (2.0).as_T()],
            [(2.0).as_T(),  (8.0).as_T()],  
            [(3.0).as_T(),  (1.0).as_T()],
            [(0.0).as_T(),  (3.0).as_T()],
        ])
    }

    fn is_descending_order<T: FloatT>(s: &[T]) -> bool {
        // is_sorted is only available post v1.82
        s.windows(2).all(|w| w[0] >= w[1])
    }


    fn run_svd_factor_test<T>(A: &mut Matrix<T>, tolfn: fn(T) -> T)
    where
        T: FloatT,
    {
        use crate::algebra::{DenseMatrix, MultiplyGEMM, VectorMath};

        let methods = [
            SVDEngineAlgorithm::DivideAndConquer,
            SVDEngineAlgorithm::QRDecomposition,
        ];

        for method in methods.iter() {

            let Acopy = A.clone(); //A is corrupted after factorization

            let mut eng = SVDEngine::<T>::new(A.size());
            eng.algorithm = *method;

            assert!(eng.factor(A).is_ok());

            let mut M = Matrix::<T>::zeros((1, 1));
            M.resize(A.size()); //manual resize for test coverage

            let U = &eng.U;
            let s = &eng.s;
            let Vt = &eng.Vt;

            assert!(is_descending_order(s));

            //reconstruct matrix from SVD
            let mut Us = U.clone();
            for c in 0..s.len() {
                for r in 0..Us.nrows() {
                    Us[(r, c)] *= s[c];
                }
            }
            M.mul(&Us, Vt, T::one(), T::zero());
            assert!(M.data().norm_inf_diff(Acopy.data()) < tolfn((1e-10).as_T()));
        }
    }


    macro_rules! generate_test_svd_factor {
        ($fxx:ty, $test_name:ident, $tolfn:ident) => {
            #[test]
            fn $test_name() {
                let mut A = test_factor_data_2x2::<$fxx>();
                run_svd_factor_test(&mut A,  |x| x.$tolfn());

                let mut A = test_factor_data_3x3::<$fxx>();  
                run_svd_factor_test(&mut A,  |x| x.$tolfn());

                let mut A = test_factor_data_4x4::<$fxx>();
                run_svd_factor_test(&mut A,  |x| x.$tolfn());

                let mut A = test_factor_data_2x4::<$fxx>();
                run_svd_factor_test(&mut A,  |x| x.$tolfn());

                let mut A = test_factor_data_4x2::<$fxx>();
                run_svd_factor_test(&mut A,  |x| x.$tolfn());
            }
        };
    }

    generate_test_svd_factor!(f32, test_svd_factor_f32, sqrt);
    generate_test_svd_factor!(f64, test_svd_factor_f64, abs);

}



#[cfg(all(test, feature = "bench"))]
mod bench {

    use super::*;

    fn svd3_bench_iter() -> impl Iterator<Item = Matrix<f64>> {

        use itertools::iproduct;

        let v = [-4., -2., 0., 1., 5.];

        iproduct!(v, v, v, v, v, v, v, v, v).map(move |(a, b, c, d, e, f, g, h, i)| {
            let data = [a,b,c,d,e,f,g,h,i];
            Matrix::new_from_slice((3,3), &data)
        })
    }

    #[test]
    fn bench_svd3_vs_blas() {

        let mut eng = SVDEngine::<f64>::new((3,3));

        for mut A in svd3_bench_iter() {
            eng.factor3(&mut A).unwrap();
        }

        for mut A in svd3_bench_iter() {
            eng.factorblas(&mut A).unwrap();
        }
    }

}

