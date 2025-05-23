use crate::algebra::{CscMatrix, FloatT};
use amd::Info;

pub mod auto;
pub mod config;
pub mod qdldl;

#[cfg(feature = "faer-sparse")]
pub mod faer_ldl;

#[cfg(any(feature = "pardiso-panua", feature = "pardiso-mkl"))]
pub mod pardiso;

#[allow(dead_code)]
pub(crate) fn amd_order<T>(KKT: &CscMatrix<T>) -> (Vec<usize>, Vec<usize>, Info)
where
    T: FloatT,
{
    // manually compute an AMD ordering for the KKT matrix
    let amd_dense_scale = 1.5; // magic number from QDLDL
    let (perm, iperm, info) = crate::qdldl::get_amd_ordering(KKT, amd_dense_scale);
    (perm, iperm, info)
}
