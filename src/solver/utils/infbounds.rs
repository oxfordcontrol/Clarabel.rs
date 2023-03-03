use crate::solver::utils::atomic::{AtomicF64, Ordering};
use crate::solver::_INFINITY_DEFAULT;
use lazy_static::lazy_static;
//
lazy_static! {
    static ref INFINITY: AtomicF64 = AtomicF64::new(_INFINITY_DEFAULT);
}

/// Revert internal infinity bound to its default value.
pub fn default_infinity() {
    INFINITY.store(_INFINITY_DEFAULT, Ordering::Relaxed);
}
/// Set the internal infinity bound to a new value.
pub fn set_infinity(v: f64) {
    INFINITY.store(v, Ordering::Relaxed);
}
/// Revert internal infinity bound to its default value.
pub fn get_infinity() -> f64 {
    INFINITY.load(Ordering::Relaxed)
}
