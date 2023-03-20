#![allow(non_snake_case)]

extern crate openblas_src;
use crate::algebra::{DenseFactorizationError, FactorEigen, FloatT, Matrix, ShapedMatrix};
use lapack::{dsyevr, ssyevr};

pub struct EigEngine<T> {
    /// Computed eigenvalues in ascending order
    pub λ: Vec<T>,

    /// Computed eigenvectors (optional)
    pub V: Option<Matrix<T>>,

    // BLAS workspace (allocated vecs only)
    isuppz: Vec<i32>,
    work: Vec<T>,
    iwork: Vec<i32>,
}

impl<T> EigEngine<T>
where
    T: FloatT,
{
    pub fn new(n: usize) -> Self {
        let λ = vec![T::zero(); n];
        let V = None;
        let isuppz = vec![0; 2 * n];
        let work = vec![T::one()];
        let iwork = vec![1];
        Self {
            λ,
            V,
            isuppz,
            work,
            iwork,
        }
    }
}

macro_rules! impl_blas_syevr {
    ($T:ty, $SYEVR:path) => {
        impl FactorEigen for EigEngine<$T> {
            type T = $T;
            fn eigvals(&mut self, A: &mut Matrix<$T>) -> Result<(), DenseFactorizationError> {
                self.syevr(A, b'N')
            }
            fn eigen(&mut self, A: &mut Matrix<$T>) -> Result<(), DenseFactorizationError> {
                self.syevr(A, b'V')
            }
        }

        impl EigEngine<$T> {
            /// Computed eigenvalues
            fn syevr(
                &mut self,
                A: &mut Matrix<$T>,
                jobz: u8,
            ) -> Result<(), DenseFactorizationError> {
                if !A.is_square() || A.nrows() != self.λ.len() {
                    return Err(DenseFactorizationError::IncompatibleDimension);
                }
                let An = A.nrows();

                // allocate for eigenvectors on first request
                if jobz == b'V' && self.V.is_none() {
                    self.V = Some(Matrix::<$T>::zeros((An, An)));
                }

                // target for computed eigenvectors (if any)
                let mut Vfake = [0 as $T];
                let Vdata = match self.V.as_mut() {
                    Some(V) => V.data_mut(),
                    None => &mut Vfake,
                };

                // standard BLAS ?symevr arguments for computing
                // a full set of eigenvalues.
                let jobz = jobz; // 'N' for values, 'V' for vecs/values
                let range = b'A'; // compute all eigenvalues
                let uplo = b'U'; // we always assume triu form
                let n = An.try_into().unwrap();
                let a = A.data_mut();
                let lda = n;
                let vl = 0.; // eig value lb (range = A => not used)
                let vu = 0.; // eig value ub (range = A => not used)
                let il = 0_i32; // eig interval lb (range = A => not used)
                let iu = 0_i32; // eig interval lb (range = A => not used)
                let abstol = -1.; // forces default tolerance
                let m = &mut 0_i32; // returns # of computed eigenvalues
                let w = &mut self.λ; // eigenvalues go here
                let z = Vdata; // vectors go here
                let ldz = n; // leading dim of eigenvector matrix
                let isuppz = &mut self.isuppz;
                let work = &mut self.work;
                let mut lwork = -1_i32; // -1 => config to request required work size
                let iwork = &mut self.iwork;
                let mut liwork = -1_i32; // -1 => config to request required work size
                let info = &mut 0_i32; // output info

                for i in 0..2 {
                    // Two calls to BLAS. First one gets sizes for work and iwork.
                    unsafe {
                        $SYEVR(
                            jobz, range, uplo, n, a, lda, vl, vu, il, iu, abstol, m, w, z, ldz,
                            isuppz, work, lwork, iwork, liwork, info,
                        );
                    }
                    if *info != 0 {
                        return Err(DenseFactorizationError::Eigen(*info));
                    }
                    // resize work vectors and reset lengths
                    if i == 0 {
                        lwork = work[0] as i32;
                        liwork = iwork[0];
                        work.resize(lwork as usize, 0.);
                        iwork.resize(liwork as usize, 0);
                    }
                }
                Ok(())
            }
        }
    };
}

impl_blas_syevr!(f32, ssyevr);
impl_blas_syevr!(f64, dsyevr);

#[test]
fn test_eigen() {
    use crate::algebra::{DenseMatrix, MultiplyGEMM, VectorMath};

    let mut S = Matrix::new_from_slice((3, 3), &[3., 2., 4., 2., 0., 2., 4., 2., 3.]);
    let Scopy = S.clone(); //S is corrupted after factorization

    let mut eng = EigEngine::<f64>::new(3);
    assert!(eng.eigvals(&mut S).is_ok());
    let sol = [-1.0, -1.0, 8.];
    assert!(eng.λ.norm_inf_diff(&sol) < 1e-6);

    let mut S = Scopy.clone(); //S is corrupted after factorization
    assert!(eng.eigen(&mut S).is_ok());
    let λ = &eng.λ;
    let mut M = Matrix::<f64>::zeros((3, 3));
    let V = eng.V.unwrap();
    let mut Vs = V.clone();
    for c in 0..3 {
        for r in 0..3 {
            Vs[(r, c)] *= λ[c];
        }
    }

    M.mul(&Vs, &V.t(), 1.0, 0.0);
    assert!(M.data().norm_inf_diff(Scopy.data()) < 1e-8);
}
