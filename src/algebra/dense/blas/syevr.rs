#![allow(non_snake_case)]
use crate::algebra::*;

pub(crate) struct EigBlasWorkVectors<T> {
    isuppz: Vec<i32>,
    work: Vec<T>,
    iwork: Vec<i32>,
}

impl<T> EigBlasWorkVectors<T>
where
    T: FloatT,
{
    fn new(n: usize) -> Self {
        let isuppz = vec![0; 2 * n];
        // must be at least 1 element because the
        // requiring work size is written into the
        // first element
        let work = vec![T::one()];
        let iwork = vec![1];
        Self {
            isuppz,
            work,
            iwork,
        }
    }
}

pub(crate) struct EigEngine<T> {
    /// Computed eigenvalues in ascending order
    pub λ: Vec<T>,

    /// Computed eigenvectors (optional)
    pub V: Option<Matrix<T>>,

    // BLAS workspace (allocated vecs only)
    pub blas: Option<EigBlasWorkVectors<T>>,
}

impl<T> EigEngine<T>
where
    T: FloatT,
{
    pub fn new(n: usize) -> Self {
        let λ = vec![T::zero(); n];
        let V = None;

        match n {
            1..=3 => Self { λ, V, blas: None },
            _ => {
                let blas = Some(EigBlasWorkVectors::new(n));
                Self { λ, V, blas }
            }
        }
    }

    pub fn n(&self) -> usize {
        self.λ.len()
    }

    fn checkdim<S>(
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
}

impl<T> FactorEigen<T> for EigEngine<T>
where
    T: FloatT,
{
    fn eigvals<S>(
        &mut self,
        A: &mut DenseStorageMatrix<S, T>,
    ) -> Result<(), DenseFactorizationError>
    where
        S: AsMut<[T]> + AsRef<[T]>,
    {
        self.checkdim(A)?;
        match self.n() {
            1 => self.eigvals1(A),
            2 => self.eigvals2(A),
            3 => self.eigvals3(A),
            _ => self.syevr(A, b'N'),
        }
    }

    #[allow(dead_code)] //for future use in projection
    fn eigen<S>(&mut self, A: &mut DenseStorageMatrix<S, T>) -> Result<(), DenseFactorizationError>
    where
        S: AsMut<[T]> + AsRef<[T]>,
    {
        self.checkdim(A)?;
        match self.n() {
            1 => self.eigen1(A),
            2 => self.eigen2(A),
            3 => self.eigen3(A),
            _ => self.syevr(A, b'V'),
        }
    }
}

// trivial implementation for 1x1 matrices

impl<T> EigEngine<T>
where
    T: FloatT,
{
    fn eigvals1<S>(
        &mut self,
        A: &mut DenseStorageMatrix<S, T>,
    ) -> Result<(), DenseFactorizationError>
    where
        S: AsMut<[T]> + AsRef<[T]>,
    {
        self.λ[0] = A[(0, 0)];
        Ok(())
    }

    fn eigen1<S>(&mut self, A: &mut DenseStorageMatrix<S, T>) -> Result<(), DenseFactorizationError>
    where
        S: AsMut<[T]> + AsRef<[T]>,
    {
        let V = self.V.get_or_insert_with(|| Matrix::<T>::zeros((1, 1)));
        self.λ[0] = A[(0, 0)];
        V[(0, 0)] = T::one();
        Ok(())
    }
}

// implementation for 2x2 matrices

impl<T> EigEngine<T>
where
    T: FloatT,
{
    fn eigvals2<S>(
        &mut self,
        A: &mut DenseStorageMatrix<S, T>,
    ) -> Result<(), DenseFactorizationError>
    where
        S: AsMut<[T]> + AsRef<[T]>,
    {
        // symmetric 2x2, stack allocated
        let mut As = DenseMatrixSym2::<T>::from(A.sym_up());
        let e = As.eigvals();
        self.λ.copy_from_slice(&e);
        Ok(())
    }

    fn eigen2<S>(&mut self, A: &mut DenseStorageMatrix<S, T>) -> Result<(), DenseFactorizationError>
    where
        S: AsMut<[T]> + AsRef<[T]>,
    {
        let Vout = self.V.get_or_insert_with(|| Matrix::<T>::zeros((2, 2)));

        // symmetric 2x2, stack allocated
        let mut As = DenseMatrixSym2::<T>::from(A.sym_up());
        let mut V = DenseMatrix2::<T>::zeros();
        let e = As.eigen(&mut V);
        self.λ.copy_from_slice(&e);
        Vout.data.copy_from(V.data());
        Ok(())
    }
}

// implementation for 3x3 matrices

impl<T> EigEngine<T>
where
    T: FloatT,
{
    fn eigvals3<S>(
        &mut self,
        A: &mut DenseStorageMatrix<S, T>,
    ) -> Result<(), DenseFactorizationError>
    where
        S: AsMut<[T]> + AsRef<[T]>,
    {
        // symmetric 3x3, stack allocated
        let mut As = DenseMatrixSym3::<T>::from(A.sym_up());
        let e = As.eigvals();
        self.λ.copy_from_slice(&e);
        Ok(())
    }

    fn eigen3<S>(&mut self, A: &mut DenseStorageMatrix<S, T>) -> Result<(), DenseFactorizationError>
    where
        S: AsMut<[T]> + AsRef<[T]>,
    {
        let Vout = self.V.get_or_insert_with(|| Matrix::<T>::zeros((3, 3)));

        // symmetric 3x3, stack allocated
        let mut As = DenseMatrixSym3::<T>::from(A.sym_up());
        let mut V = DenseMatrix3::<T>::zeros();
        let e = As.eigen(&mut V);
        self.λ.copy_from_slice(&e);
        Vout.data.copy_from(V.data());
        Ok(())
    }
}

// implementation for arbitrary size matrices

impl<T> EigEngine<T>
where
    T: FloatT,
{
    fn syevr<S>(
        &mut self,
        A: &mut DenseStorageMatrix<S, T>,
        jobz: u8,
    ) -> Result<(), DenseFactorizationError>
    where
        S: AsMut<[T]> + AsRef<[T]>,
    {
        let An = self.n();

        // allocate for eigenvectors on first request
        if jobz == b'V' && self.V.is_none() {
            self.V = Some(Matrix::<T>::zeros((An, An)));
        }

        // unwrap or populate on the first call
        let blaswork = self.blas.get_or_insert_with(|| EigBlasWorkVectors::new(An));

        // standard BLAS ?syevr arguments for computing a full set of eigenvalues.

        let range = b'A'; // compute all eigenvalues
        let uplo = MatrixTriangle::Triu.as_blas_char(); // we always assume triu form
        let n = An.try_into().unwrap();
        let a = A.data_mut();
        let lda = n;
        let vl = T::zero(); // eig value lb (range = A => not used)
        let vu = T::zero(); // eig value ub (range = A => not used)
        let il = 0_i32; // eig interval lb (range = A => not used)
        let iu = 0_i32; // eig interval ub (range = A => not used)
        let abstol = -T::one(); // forces default tolerance
        let m = &mut 0_i32; // returns # of computed eigenvalues
        let w = &mut self.λ; // eigenvalues go here
        let ldz = n; // leading dim of eigenvector matrix
        let isuppz = &mut blaswork.isuppz;
        let work = &mut blaswork.work;
        let mut lwork = -1_i32; // -1 => config to request required work size
        let iwork = &mut blaswork.iwork;
        let mut liwork = -1_i32; // -1 => config to request required work size
        let info = &mut 0_i32; // output info

        // target for computed eigenvectors (if any)
        let mut tmpz = [T::zero()]; //placeholder if V empty
        let z = match self.V.as_mut() {
            Some(V) => V.data_mut(),
            None => tmpz.as_mut_slice(), // fake target
        };

        for i in 0..2 {
            T::xsyevr(
                jobz, range, uplo, n, a, lda, vl, vu, il, iu, abstol, m, w, z, ldz, isuppz, work,
                lwork, iwork, liwork, info,
            );
            if *info != 0 {
                return Err(DenseFactorizationError::Eigen(*info));
            }
            // resize work vectors and reset lengths
            if i == 0 {
                lwork = work[0].to_i32().unwrap();
                liwork = iwork[0];
                work.resize(lwork as usize, T::zero());
                iwork.resize(liwork as usize, 0);
            }
        }
        Ok(())
    }
}

macro_rules! generate_test_eigen {
    ($fxx:ty, $test_name:ident, $tolfn:ident) => {
        #[test]
        fn $test_name() {
            use crate::algebra::{DenseMatrix, MultiplyGEMM, VectorMath};

            // has to be 4x4 to avoid the special case
            let mut S = Matrix::<$fxx>::from(&[
                [3., 2., 4., 0.], //
                [2., 0., 2., 0.], //
                [4., 2., 3., 0.], //
                [0., 0., 0., 9.], //
            ]);

            let Scopy = S.clone(); //S is corrupted after factorization

            let mut eng = EigEngine::<$fxx>::new(4);
            assert!(eng.eigvals(&mut S).is_ok());
            let sol = [-1.0, -1.0, 8., 9.];
            assert!(eng.λ.norm_inf_diff(&sol) < 1e-6);

            let mut S = Scopy.clone(); //S is corrupted after factorization
            assert!(eng.eigen(&mut S).is_ok());
            let λ = &eng.λ;
            let mut M = Matrix::<$fxx>::zeros((4, 4));
            let V = eng.V.unwrap();
            let mut Vs = V.clone();
            for c in 0..4 {
                for r in 0..4 {
                    Vs[(r, c)] *= λ[c];
                }
            }

            M.mul(&Vs, &V.t(), 1.0, 0.0);
            assert!(M.data().norm_inf_diff(Scopy.data()) < (1e-11 as $fxx).$tolfn());
        }
    };
}

generate_test_eigen!(f32, test_eigen_f32, sqrt);
generate_test_eigen!(f64, test_eigen_f64, abs);

#[cfg(test)]
mod tests {
    use super::*;
    // minimal test for debugging
    #[test]
    fn test_eig2() {
        let mut S = Matrix::<f64>::from(&[
            [3., 2.], //
            [2., 0.], //
        ]);

        let mut eng = EigEngine::<f64>::new(2);
        assert!(eng.eigen(&mut S).is_ok());
    }
}

#[cfg(all(test, feature = "bench"))]
mod bench {

    use super::*;

    fn eig3_bench_iter() -> impl Iterator<Item = Matrix<f64>> {
        use itertools::iproduct;

        let v = [-4., -2., 0., 1., 3.14, 5., 12.];

        iproduct!(v, v, v, v, v, v).map(move |(a, b, c, d, e, f)| {
            let data = [a, b, c, 0., d, e, 0., 0., f];
            Matrix::new_from_slice((3, 3), &data)
        })
    }

    #[test]
    fn bench_eig3_vs_blas() {
        let mut eng = EigEngine::<f64>::new(3);

        for mut A in eig3_bench_iter() {
            let _ = eng.syevr(&mut A, b'N');
        }

        for mut A in eig3_bench_iter() {
            let _ = eng.eigvals3(&mut A);
        }
    }
}
