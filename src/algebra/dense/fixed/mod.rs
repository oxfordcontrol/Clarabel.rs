// Only the 3x3 cholesky factorisation is required for
// core solver.  Everything else is required for dense
// operations supporting SDPs.

mod types;
pub(crate) use self::types::*;
mod dense3x3;
pub(crate) use self::dense3x3::*;

// these only required for factorization engine
// objects that support SDPs
cfg_if::cfg_if! {
    if #[cfg(feature = "sdp")] {
        mod dense2x2;
        pub(crate) use self::dense2x2::*;
    }
}
