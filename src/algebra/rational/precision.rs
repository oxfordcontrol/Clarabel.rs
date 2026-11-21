//! Thread-local precision settings for [`RationalReal`].
//!
//! Two independent knobs:
//!
//! - **Working precision** (`precision_bits`): used by the transcendentals
//!   (`sqrt`, `ln`, `exp`, `powf`) as the convergence tolerance and the
//!   denominator cap they round their result to. Default 128 bits.
//!
//! - **Max arena bits** (`max_arena_bits`): if `Some(p)`, every arithmetic
//!   op (`+`, `-`, `*`, `/`, `-`, `%`) rounds its result to `m / 2^p`
//!   when either the numerator or denominator exceeds `p` bits. This
//!   bounds the cost of subsequent operations at the price of giving
//!   up bit-exactness. Default `None` (exact mode).
//!
//! # Why two knobs?
//!
//! The default exact mode preserves the headline guarantee that
//! `(1/3) + (1/3) + (1/3) == 1` exactly — useful for certifying LP/QP
//! solutions or for QOU's bit-exact knot-certificate use case.
//!
//! For practical solver runs on non-trivial problems, the per-iteration
//! BigRational denominator growth makes exact mode impractical
//! (geometric blowup in denominator bits → cubic-or-worse total cost).
//! Setting `max_arena_bits(256)` keeps every intermediate at ≤256 bits
//! ≈ 77 decimal digits — still ~5× the precision of `f64` — at near-f64
//! speed. QOU's R5_FULL_PLAN.md target of ≥50 dps is satisfied at 167 bits.

use std::cell::Cell;

const DEFAULT_PRECISION_BITS: u32 = 128;

thread_local! {
    static PRECISION_BITS: Cell<u32> = const { Cell::new(DEFAULT_PRECISION_BITS) };
    static MAX_ARENA_BITS: Cell<Option<u32>> = const { Cell::new(None) };
}

/// Get the current working precision in bits for transcendentals on this
/// thread. Default is 128 bits.
pub fn precision_bits() -> u32 {
    PRECISION_BITS.with(|p| p.get())
}

/// Set the working precision in bits for transcendentals on this thread.
/// Returns the previous value so callers can restore it (use
/// [`with_precision`] for a scope-guarded version).
///
/// Panics if `bits == 0`.
pub fn set_precision_bits(bits: u32) -> u32 {
    assert!(bits > 0, "RationalReal precision must be positive");
    PRECISION_BITS.with(|p| p.replace(bits))
}

/// Run a closure with the working precision temporarily set to `bits`,
/// restoring the previous value on return. Resets unconditionally even on
/// panic (the closure runs in a guard).
pub fn with_precision<R>(bits: u32, f: impl FnOnce() -> R) -> R {
    struct Guard(u32);
    impl Drop for Guard {
        fn drop(&mut self) {
            PRECISION_BITS.with(|p| p.set(self.0));
        }
    }
    let prev = set_precision_bits(bits);
    let _guard = Guard(prev);
    f()
}

/// Get the current max-arena-bits cap. `None` means "exact mode": every
/// arithmetic op preserves the full BigRational result without rounding.
/// `Some(p)` means: results are rounded to `m / 2^p` when either
/// numerator or denominator would otherwise exceed `p` bits.
pub fn max_arena_bits() -> Option<u32> {
    MAX_ARENA_BITS.with(|c| c.get())
}

/// Set the max-arena-bits cap on this thread. Returns the previous value.
///
/// `None` is exact mode (default). `Some(p)` rounds every arithmetic
/// result to fit in `p` bits of denominator + numerator — losing
/// bit-exactness in exchange for bounded per-op runtime.
///
/// Recommended values:
/// - `None` for tiny LP/QPs where bit-exactness matters.
/// - `Some(256)` for solver runs where ~77 decimal digits is enough
///   (5× f64 precision, runs in seconds rather than minutes).
/// - `Some(167)` for QOU's R5_FULL_PLAN.md target of ≥50 dps.
///
/// Panics if `bits == Some(0)`.
pub fn set_max_arena_bits(bits: Option<u32>) -> Option<u32> {
    if let Some(b) = bits {
        assert!(b > 0, "max_arena_bits must be positive when Some");
    }
    MAX_ARENA_BITS.with(|c| c.replace(bits))
}

/// Scope-guarded version of [`set_max_arena_bits`].
pub fn with_max_arena_bits<R>(bits: Option<u32>, f: impl FnOnce() -> R) -> R {
    struct Guard(Option<u32>);
    impl Drop for Guard {
        fn drop(&mut self) {
            MAX_ARENA_BITS.with(|c| c.set(self.0));
        }
    }
    let prev = set_max_arena_bits(bits);
    let _guard = Guard(prev);
    f()
}
