//! Thread-local arena that owns the `BigRational` values referenced by
//! [`RationalReal`](super::real::RationalReal) handles.
//!
//! Design: `RationalReal` is a `Copy`-able 4-byte handle (`u32`). The
//! **top two bits** of the handle are a tag distinguishing finite values
//! from the IEEE-style sentinels (+inf / −inf / NaN); the remaining
//! 30 bits are the arena index for finite handles. Every arithmetic op
//! either propagates a sentinel tag without touching the arena, or
//! pushes a freshly-computed `BigRational` and returns a fresh
//! finite-tagged handle. The arena grows monotonically during a solve
//! and is reset between solves (or explicitly via [`reset_arena`]).
//!
//! See `src/algebra/rational/mod.rs` for the higher-level rationale and
//! the `Send + Sync` invariants.

use num_bigint::BigInt;
use num_rational::BigRational;
use std::cell::RefCell;

// =====================================================================
// Tag-bit layout for the u32 handle.
// =====================================================================
//
// bits 31..30 : tag  (00 = finite, 01 = +inf, 10 = -inf, 11 = NaN)
// bits 29..0  : arena index (only meaningful when tag = 00)
//
// 30 index bits ⇒ up to 2^30 = 1_073_741_824 finite handles per thread,
// which at the smallest plausible BigRational footprint (~64 B) is still
// ≥ 64 GB of arena before exhaustion. Substantially in excess of any
// realistic single-solve working set; in practice users will reset the
// arena between solves long before hitting this limit.

/// Mask isolating the 2-bit tag in the high bits of a handle.
pub(crate) const TAG_MASK: u32 = 0b11 << 30;
/// Mask isolating the 30-bit arena index in the low bits of a handle.
pub(crate) const INDEX_MASK: u32 = !TAG_MASK;

/// Tag bits (in their high-bit position) for the finite class.
pub(crate) const TAG_FINITE: u32 = 0b00 << 30;
/// Tag bits for the +infinity sentinel.
pub(crate) const TAG_POS_INF: u32 = 0b01 << 30;
/// Tag bits for the -infinity sentinel.
pub(crate) const TAG_NEG_INF: u32 = 0b10 << 30;
/// Tag bits for the NaN sentinel.
pub(crate) const TAG_NAN: u32 = 0b11 << 30;

/// Sentinel handle for +infinity (tag bits only; index part is unused).
pub(crate) const POS_INF_HANDLE: u32 = TAG_POS_INF;
/// Sentinel handle for -infinity (tag bits only; index part is unused).
pub(crate) const NEG_INF_HANDLE: u32 = TAG_NEG_INF;
/// Sentinel handle for NaN (tag bits only; index part is unused).
pub(crate) const NAN_HANDLE: u32 = TAG_NAN;

/// Maximum number of finite arena entries per thread (`2^30`).
pub(crate) const MAX_ARENA_LEN: usize = 1usize << 30;

/// Returns true if `handle` is a finite-tagged handle (i.e. the top two
/// bits are `00` and the low 30 bits are a valid arena index).
#[inline]
pub(crate) fn is_finite_handle(handle: u32) -> bool {
    (handle & TAG_MASK) == TAG_FINITE
}

/// Returns true if `handle` is the +infinity sentinel.
#[inline]
pub(crate) fn is_pos_inf_handle(handle: u32) -> bool {
    (handle & TAG_MASK) == TAG_POS_INF
}

/// Returns true if `handle` is the -infinity sentinel.
#[inline]
pub(crate) fn is_neg_inf_handle(handle: u32) -> bool {
    (handle & TAG_MASK) == TAG_NEG_INF
}

/// Returns true if `handle` is the NaN sentinel.
#[inline]
pub(crate) fn is_nan_handle(handle: u32) -> bool {
    (handle & TAG_MASK) == TAG_NAN
}

/// Returns true if `handle` is one of the infinity sentinels.
#[inline]
pub(crate) fn is_infinite_handle(handle: u32) -> bool {
    let t = handle & TAG_MASK;
    t == TAG_POS_INF || t == TAG_NEG_INF
}

/// Returns true if `handle` is any of the non-finite sentinels.
#[inline]
#[allow(dead_code)] // exposed as a helper for future arena callers
pub(crate) fn is_sentinel_handle(handle: u32) -> bool {
    (handle & TAG_MASK) != TAG_FINITE
}

/// Extract the 30-bit arena index from a finite handle. The result is
/// only meaningful when `is_finite_handle(handle)` is true; for sentinel
/// handles the returned value has no semantic meaning.
#[inline]
pub(crate) fn index_of(handle: u32) -> u32 {
    handle & INDEX_MASK
}

/// Construct a finite handle from a 30-bit arena index. Debug builds
/// assert that `idx` fits in 30 bits.
#[inline]
pub(crate) fn finite_handle(idx: u32) -> u32 {
    debug_assert!(idx <= INDEX_MASK, "arena index does not fit in 30 bits");
    TAG_FINITE | (idx & INDEX_MASK)
}

thread_local! {
    /// Per-thread storage. `Vec` (not `HashMap`) because handles are dense
    /// `u32` indices so a Vec lookup is O(1) with no hashing.
    static ARENA: RefCell<Vec<BigRational>> = const { RefCell::new(Vec::new()) };
    /// Per-thread monotonic counter; bumped by every [`reset_arena`] call.
    /// Caches that hold arena handles (sentinel, const, ...) snapshot this
    /// value at populate time and re-derive when it changes — robust even
    /// if the arena is reset and immediately repopulated by other code.
    static ARENA_GEN: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

/// Push a `BigRational` into the thread-local arena and return a
/// finite-tagged handle.
///
/// Panics on overflow if the arena would exceed [`MAX_ARENA_LEN`]
/// (`2^30`) entries — at that point a `reset_arena()` is overdue.
#[inline]
pub(crate) fn push(value: BigRational) -> u32 {
    ARENA.with(|cell| {
        let mut a = cell.borrow_mut();
        let idx = a.len();
        if idx >= MAX_ARENA_LEN {
            panic!(
                "RationalReal arena exhausted ({} entries; cap is 2^30 = {} \
                 because the top two bits of the handle encode the \
                 finite/+inf/-inf/NaN tag). Call \
                 clarabel::algebra::rational::reset_arena() between solves.",
                idx, MAX_ARENA_LEN
            );
        }
        a.push(value);
        finite_handle(idx as u32)
    })
}

/// Read a clone of the value addressed by `handle`.
///
/// Panics if `handle` is one of the non-finite sentinels (+inf / -inf /
/// NaN); callers must dispatch on the tag bits before invoking this.
#[inline]
pub(crate) fn get(handle: u32) -> BigRational {
    debug_assert!(
        is_finite_handle(handle),
        "arena::get called with a sentinel handle (tag = {:02b}); callers must \
         check is_finite_handle / is_sentinel_handle first",
        (handle & TAG_MASK) >> 30
    );
    ARENA.with(|cell| cell.borrow()[index_of(handle) as usize].clone())
}

/// Apply a closure to a borrow of the value addressed by `handle`
/// without cloning. Sentinel handles must be filtered out by the
/// caller (see [`get`]).
#[inline]
pub(crate) fn with<R>(handle: u32, f: impl FnOnce(&BigRational) -> R) -> R {
    debug_assert!(
        is_finite_handle(handle),
        "arena::with called with a sentinel handle (tag = {:02b}); callers must \
         check is_finite_handle / is_sentinel_handle first",
        (handle & TAG_MASK) >> 30
    );
    ARENA.with(|cell| f(&cell.borrow()[index_of(handle) as usize]))
}

/// Apply a closure to two arena borrows at once. Both handles must be
/// finite; arithmetic operators dispatch on the sentinel tags first.
#[inline]
pub(crate) fn with2<R>(a: u32, b: u32, f: impl FnOnce(&BigRational, &BigRational) -> R) -> R {
    debug_assert!(
        is_finite_handle(a) && is_finite_handle(b),
        "arena::with2 called with a sentinel handle; callers must dispatch on \
         the tag bits before reaching this point"
    );
    ARENA.with(|cell| {
        let arena = cell.borrow();
        f(&arena[index_of(a) as usize], &arena[index_of(b) as usize])
    })
}

/// Reset the thread-local arena to empty. All outstanding `RationalReal`
/// handles on this thread become invalid; using one after `reset_arena()`
/// is a panic in debug builds (out-of-bounds index) or undefined behaviour
/// in release builds (unrelated `BigRational` returned at the recycled slot).
///
/// Also bumps the arena-generation counter so that handle caches
/// (sentinel, const, ...) see the change and re-derive their entries.
///
/// Call this between solves to recover memory.
pub fn reset_arena() {
    ARENA.with(|cell| cell.borrow_mut().clear());
    ARENA_GEN.with(|g| g.set(g.get().wrapping_add(1)));
}

/// Current arena generation — incremented by every [`reset_arena`] call.
/// Caches that hold handles store this at populate time and re-derive
/// when it changes.
#[inline]
pub(crate) fn arena_generation() -> u64 {
    ARENA_GEN.with(|g| g.get())
}

/// Current arena size, in entries (not bytes). Useful for diagnostics
/// and the per-iteration denominator-bit-length log.
pub fn arena_len() -> usize {
    ARENA.with(|cell| cell.borrow().len())
}

/// Convenience: push the constant zero.
#[inline]
pub(crate) fn push_zero() -> u32 {
    push(BigRational::new(BigInt::from(0), BigInt::from(1)))
}

/// Convenience: push the constant one.
#[inline]
pub(crate) fn push_one() -> u32 {
    push(BigRational::new(BigInt::from(1), BigInt::from(1)))
}
