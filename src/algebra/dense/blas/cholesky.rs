#![allow(non_snake_case)]

use crate::algebra::*;

pub(crate) struct CholeskyEngine<T> {
    /// lower triangular factor (stored as square dense)
    pub L: Matrix<T>,
}

impl<T> CholeskyEngine<T>
where
    T: FloatT,
{
    pub fn new(n: usize) -> Self {
        let L = Matrix::<T>::zeros((n, n));
        Self { L }
    }

    pub fn resize(&mut self, n: usize) {
        self.L.resize((n, n));
    }

    pub fn n(&self) -> usize {
        self.L.nrows()
    }

    fn checkdim_factor<S>(
        &mut self,
        A: &mut DenseStorageMatrix<S, T>,
    ) -> Result<(), DenseFactorizationError>
    where
        S: AsMut<[T]> + AsRef<[T]>,
    {
        if !A.is_square() || A.nrows() != self.n() {
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
        if B.nrows() != self.n() {
            Err(DenseFactorizationError::IncompatibleDimension)
        } else {
            Ok(())
        }
    }
}

impl<T> FactorCholesky<T> for CholeskyEngine<T>
where
    T: FloatT,
{
    fn factor<S>(&mut self, A: &mut DenseStorageMatrix<S, T>) -> Result<(), DenseFactorizationError>
    where
        S: AsMut<[T]> + AsRef<[T]>,
    {
        self.checkdim_factor(A)?;
        match self.n() {
            1 => self.factor1(A),
            2 => self.factor2(A),
            3 => self.factor3(A),
            _ => self.factorblas(A),
        }
    }

    fn solve<S>(&mut self, B: &mut DenseStorageMatrix<S, T>)
    where
        S: AsMut<[T]> + AsRef<[T]>,
    {
        self.checkdim_solve(B).unwrap();
        match self.n() {
            1 => self.solve1(B),
            2 => self.solve2(B),
            3 => self.solve3(B),
            _ => self.solveblas(B),
        }
    }

    fn logdet(&self) -> T {
        let mut ld = T::zero();
        let n = self.L.nrows();
        for i in 0..n {
            ld += T::ln(self.L[(i, i)]);
        }
        ld + ld
    }
}

// trivial implementation for 1x1 matrices

impl<T> CholeskyEngine<T>
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
        let A = A.data()[0];
        if A <= T::zero() {
            // check for positive definite
            Err(DenseFactorizationError::Cholesky(1))
        }
        else {
            self.L[(0, 0)] = A.sqrt();
            Ok(())
        }
    }

    fn solve1<S>(&mut self, B: &mut DenseStorageMatrix<S, T>)
    where
        S: AsMut<[T]> + AsRef<[T]>,
    {
        let L = self.L.data()[0];
        let A = L * L;

        for col in 0..B.ncols() {
            let b = B.col_slice_mut(col);
            b[0] /= A;
        }
    }
}

// implementation for 2x2 matrices

impl<T> CholeskyEngine<T>
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
        // symmetric 3x3, stack allocated
        let As = DenseMatrixSym2::<T>::from(A.sym_up());
        let mut L = DenseMatrixSym2::<T>::zeros();

        L.cholesky_2x2_explicit_factor(&As)?;

        // push the result into our internal factor matrix
        // internal representation is lower triangular
        self.L[(0, 0)] = L.data[0];
        self.L[(1, 0)] = L.data[1];
        self.L[(1, 1)] = L.data[2];

        // check for positive definite
        Ok(())
    }

    fn solve2<S>(&mut self, B: &mut DenseStorageMatrix<S, T>)
    where
        S: AsMut<[T]> + AsRef<[T]>,
    {
        let L = DenseMatrixSym2::<T>::from(self.L.sym_lo());
        let mut x = [T::zero(); 2];

        for col in 0..B.ncols() {
            let b = B.col_slice_mut(col);

            // solve for x
            L.cholesky_2x2_explicit_solve(&mut x, b);
            // expected behaviour is to solve in place
            b.copy_from_slice(&x);
        }
    }
}

// implementation for 3x3 matrices

impl<T> CholeskyEngine<T>
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
        // symmetric 3x3, stack allocated
        let As = DenseMatrixSym3::<T>::from(A.sym_up());
        let mut L = DenseMatrixSym3::<T>::zeros();

        L.cholesky_3x3_explicit_factor(&As)?;

        // push the result into our internal factor matrix
        // internal representation is lower triangular
        self.L[(0, 0)] = L.data[0];
        self.L[(1, 0)] = L.data[1];
        self.L[(2, 0)] = L.data[3];
        self.L[(1, 1)] = L.data[2];
        self.L[(2, 1)] = L.data[4];
        self.L[(2, 2)] = L.data[5];

        // check for positive definite
        Ok(())
    }

    fn solve3<S>(&mut self, B: &mut DenseStorageMatrix<S, T>)
    where
        S: AsMut<[T]> + AsRef<[T]>,
    {
        let L = DenseMatrixSym3::<T>::from(self.L.sym_lo());
        let mut x = [T::zero(); 3];

        for col in 0..B.ncols() {
            let b = B.col_slice_mut(col);

            // solve for x
            L.cholesky_3x3_explicit_solve(&mut x, b);
            // expected behaviour is to solve in place
            b.copy_from_slice(&x);
        }
    }
}

// implementation for arbitrary size matrices

impl<T> CholeskyEngine<T>
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
        // ?potrf factors in place, so first copy A onto
        // our internal factor matrix L.  We reference the
        // upper triangle of A, but want a lower triangular
        // result.  LAPACK factors triu inputs to U^TU, and
        // tril inputs to LL^T, so first copy the triu part
        // of A into tril of L before we factor it
        let At = A.t();
        let n = self.L.nrows();
        for j in 0..n {
            for i in j..n {
                self.L[(i, j)] = At[(i, j)];
            }
        }

        // standard BLAS ?potrf arguments for computing
        // cholesky decomposition
        let uplo = MatrixTriangle::Tril.as_blas_char();
        let An = self.L.nrows().try_into().unwrap();
        let a = self.L.data_mut();
        let lda = An;
        let info = &mut 0_i32; // output info

        T::xpotrf(uplo, An, a, lda, info);

        if *info != 0 {
            return Err(DenseFactorizationError::Cholesky(*info));
        }

        Ok(())
    }

    fn solveblas<S>(&mut self, B: &mut DenseStorageMatrix<S, T>)
    where
        S: AsMut<[T]> + AsRef<[T]>,
    {
        // standard BLAS ?potrs arguments for computing
        // post factorization triangular solve

        // Tril here since we transposed A into L before
        // factoring it
        let uplo = MatrixTriangle::Tril.as_blas_char();

        let nrhs = B.ncols().try_into().unwrap();
        let An = self.L.nrows().try_into().unwrap();
        let a = &self.L.data;
        let lda = An;
        let Bn = B.nrows().try_into().unwrap();
        let b = B.data_mut();
        let ldb = Bn;
        let info = &mut 0_i32; // output info

        T::xpotrs(uplo, An, nrhs, a, lda, b, ldb, info);

        assert_eq!(*info, 0);
    }
}

// ---- unit testing ----

#[cfg(test)]
mod test {

    use super::*;

    fn test_data_2x2<T: FloatT>() -> (Matrix<T>, Matrix<T>, Matrix<T>) {
        // Create a symmetric matrix S
        let S = Matrix::<T>::from(&[
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
    
        (S, X, B)
    }
 
    #[rustfmt::skip]
    fn test_data_3x3<T: FloatT>() -> (Matrix<T>, Matrix<T>, Matrix<T>) {
        let S = Matrix::<T>::from(&[
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

        (S, X, B)
    }

       
    #[rustfmt::skip]
    fn test_data_4x4<T: FloatT>() -> (Matrix<T>, Matrix<T>, Matrix<T>) {
        // Create a symmetric matrix S
        let S = Matrix::<T>::from(&[
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

        (S, X, B)
    }

    fn run_test<T>(S: &mut Matrix<T>, X: &Matrix<T>, B: &mut Matrix<T>, tolfn: fn(T) -> T)
    where
        T: FloatT,
    {
        use crate::algebra::{DenseMatrix, MultiplyGEMM, VectorMath};

        let n = S.nrows();
        let Scopy = S.clone(); //S is corrupted after factorization

        let mut eng = CholeskyEngine::<T>::new(n);
        assert!(eng.factor(S).is_ok());

        let mut M = Matrix::<T>::zeros((n, n));
        M.mul(&eng.L, &eng.L.t(), T::one(), T::zero());
        assert!(M.data().norm_inf_diff(Scopy.data()) < tolfn((1e-8).as_T()));
        eng.solve(B);

        assert!(B.data.norm_inf_diff(X.data()) <= tolfn((1e-12).as_T()));
    }

    macro_rules! generate_test_cholesky {
        ($fxx:ty, $test_name:ident, $tolfn:ident) => {
            #[test]
            fn $test_name() {
                let (mut S, X, mut B) = test_data_2x2::<$fxx>();
                run_test(&mut S, &X, &mut B, |x| x.$tolfn());

                let (mut S, X, mut B) = test_data_3x3::<$fxx>();
                run_test(&mut S, &X, &mut B, |x| x.$tolfn());

                let (mut S, X, mut B) = test_data_4x4::<$fxx>();
                run_test(&mut S, &X, &mut B, |x| x.$tolfn());
            }
        };
    }

    generate_test_cholesky!(f32, test_cholesky_f32, sqrt);
    generate_test_cholesky!(f64, test_cholesky_f64, abs);

    macro_rules! generate_test_cholesky_logdet {
        ($fxx:ty, $test_name:ident, $tolfn:ident) => {
            #[test]
            #[allow(clippy::excessive_precision)]
            fn $test_name() {
                #[rustfmt::skip]
            let mut S = Matrix::<$fxx>::from(
            &[[ 8., -2., 4.],
              [-2., 12., 2.],
              [ 4.,  2., 6.]]);

                let mut eng = CholeskyEngine::<$fxx>::new(3);
                assert!(eng.factor(&mut S).is_ok());
                assert!((eng.logdet() - 5.69035945432406).abs() < (1e-10 as $fxx).$tolfn());
            }
        };
    }

    generate_test_cholesky_logdet!(f32, test_cholesky_logdet_f32, sqrt);
    generate_test_cholesky_logdet!(f64, test_cholesky_logdet_f64, abs);
}




#[cfg(all(test, feature = "bench"))]
mod bench {

    use super::*;

    fn cholesky3_bench_iter() -> impl Iterator<Item = Matrix<f64>> {
        use itertools::iproduct;

        let v: Vec<f64> = (-100..=100).map(|i| i as f64).collect();

        iproduct!(v.clone(), v.clone(), v.clone()).map(move |(b,d,e)| {
            // new matrices that are positive definite, 
            // so choose a,c,e so that the matrix is 
            // diagonally dominant
            let a = b.abs() + d.abs() + 0.1;
            let c = b.abs() + e.abs() + 0.1;
            let f = d.abs() + e.abs() + 0.1;
            let data = [a,b,d,b,c,e,d,e,f];
            Matrix::new_from_slice((3,3), &data)
        })
    }

    #[test]
    fn bench_cholesky3_vs_blas() {
        let mut eng = CholeskyEngine::<f64>::new(3);

        for mut A in cholesky3_bench_iter() {
            let _ = eng.factor3(&mut A);
        }

        for mut A in cholesky3_bench_iter() {
            let _ = eng.factorblas(&mut A);
        }
    }
}