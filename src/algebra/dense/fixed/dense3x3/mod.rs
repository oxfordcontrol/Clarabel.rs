// Only the 3x3 cholesky factorisation is required for
// core solver.  Everything else is required for dense
// operations supporting SDPs.

// Make cholesky module private to parent
mod cholesky;
mod core;

pub(crate) use self::core::*;

// these only required for factorization engine
// objects that support SDPs
cfg_if::cfg_if! {
    if #[cfg(feature = "sdp")] {
        pub(crate) mod eigen;
        pub(crate) mod svd;
    }
}
