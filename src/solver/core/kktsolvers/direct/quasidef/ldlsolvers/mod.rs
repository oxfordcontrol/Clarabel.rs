use crate::algebra::{CscMatrix, FloatT};
use amd::Info;

#[cfg(feature = "faer-sparse")]
pub mod faer_ldl;

pub mod auto;
pub mod qdldl;

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
