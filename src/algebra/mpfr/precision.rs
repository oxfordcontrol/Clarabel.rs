//! Thread-local default precision for new [`MpfrFloat`] values.
//!
//! `rug::Float` carries its own per-value precision (in bits). When we
//! construct an `MpfrFloat` from a primitive (`from_i64`, `from_f64`,
//! `zero`, etc.), we use the thread-local default set here. Arithmetic
//! between two `MpfrFloat`s uses the larger of their precisions.
//!
//! Default: 167 bits ≈ 50 decimal digits. Matches QOU's
//! R5_FULL_PLAN.md target precision.

use std::cell::Cell;

const DEFAULT_PRECISION_BITS: u32 = 167;

thread_local! {
    static DEFAULT_PRECISION: Cell<u32> = const { Cell::new(DEFAULT_PRECISION_BITS) };
}

/// Get the current default MPFR precision in bits for new
/// [`MpfrFloat`](super::real::MpfrFloat) values on this thread.
pub fn default_precision() -> u32 {
    DEFAULT_PRECISION.with(|p| p.get())
}

/// Set the default MPFR precision in bits for new `MpfrFloat` values
/// on this thread. Returns the previous value. Existing `MpfrFloat`
/// values keep their construction-time precision; only newly-constructed
/// values pick up the new default.
///
/// Panics if `bits == 0` (rug requires positive precision).
pub fn set_default_precision(bits: u32) -> u32 {
    assert!(bits > 0, "MpfrFloat precision must be positive");
    DEFAULT_PRECISION.with(|p| p.replace(bits))
}

/// Run a closure with the default MPFR precision temporarily set to
/// `bits`, restoring the previous value on return (panic-safe via
/// a `Drop` guard).
pub fn with_precision<R>(bits: u32, f: impl FnOnce() -> R) -> R {
    struct Guard(u32);
    impl Drop for Guard {
        fn drop(&mut self) {
            DEFAULT_PRECISION.with(|p| p.set(self.0));
        }
    }
    let prev = set_default_precision(bits);
    let _guard = Guard(prev);
    f()
}
