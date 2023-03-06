use crate::solver::utils::atomic::{AtomicF64, Ordering};
use lazy_static::lazy_static;

/// Constant indicating that an inequality bound is to be treated as infinite.
///   
/// If the setting [`presolve_enable`](crate::solver::DefaultSettings::presolve_enable)
/// is `true`, any such constraints are removed.   Bounds for all other cones with
/// values greather than this are capped at this value.
/// A custom constant for this bound can be specified using [`set_infinity`].  
///
/// Setting the infinity bound to a custom constant applies at module level.
///
pub const INFINITY_DEFAULT: f64 = crate::solver::_INFINITY_DEFAULT;

lazy_static! {
    static ref INFINITY: AtomicF64 = AtomicF64::new(INFINITY_DEFAULT);
}

/// Revert internal infinity bound to its default value.   The default is [`INFINITY_DEFAULT`]
///
/// See also: [`get_infinity`], [`set_infinity`]
pub fn default_infinity() {
    INFINITY.store(INFINITY_DEFAULT, Ordering::Relaxed);
}
/// Set the internal infinity bound to a new value.
///
/// See also: [`get_infinity`], [`default_infinity`]
pub fn set_infinity(v: f64) {
    INFINITY.store(v, Ordering::Relaxed);
}
/// Get the current value of the internal infinity bound.
///
/// See also: [`set_infinity`], [`default_infinity`]
pub fn get_infinity() -> f64 {
    INFINITY.load(Ordering::Relaxed)
}
