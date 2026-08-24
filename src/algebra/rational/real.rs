//! [`RationalReal`] — Copy-able arena handle that satisfies `FloatT`.
//!
//! See `super` (mod.rs) for the design rationale and Send + Sync invariant.
//!
//! # Handle layout
//!
//! `RationalReal` wraps a single `u32` whose top two bits encode the
//! IEEE-style class of the value (finite / +inf / -inf / NaN) and whose
//! bottom 30 bits hold the per-thread arena index when the class is
//! finite. See [`crate::algebra::rational::arena`] for the constants
//! and the propagation rules in this file's arithmetic impls.

use super::arena;
use super::cap::maybe_cap;
use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{FromPrimitive, Num, One, Signed, Zero};
use std::cmp::Ordering;
use std::ops::{
    Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign,
};

/// Exact-rational scalar handle.
///
/// 4-byte `Copy` value whose top two bits hold an IEEE-class tag
/// (finite / +inf / -inf / NaN) and whose bottom 30 bits index into a
/// thread-local arena that owns the `BigRational` value (when the tag
/// is finite). See the [`rational`](super) module docstring for the
/// memory model and invariants.
///
/// # Lifecycle
/// Construct via [`RationalReal::from_bigrational`],
/// [`RationalReal::from_pair`], or the `From<f64>`/`From<i64>` impls.
/// Use arithmetic, comparisons, and conversions inline. Extract a
/// final value via [`into_pair`](Self::into_pair) (consuming
/// `(BigInt, BigInt)`), [`to_bigrational`](Self::to_bigrational)
/// (cloning), or [`to_f64`](Self::to_f64) (lossy). Sentinel values
/// have no `BigRational` representation; calling [`to_bigrational`]
/// or [`into_pair`] on a sentinel panics — guard with
/// [`crate::algebra::transcendental::RealSentinel::is_finite`] first.
///
/// # Send + Sync
/// `RationalReal` is declared `Send + Sync` via `unsafe impl` because
/// `FloatT` requires it. The runtime invariant: a finite-tagged
/// `RationalReal` may only be dereferenced on the thread that produced
/// it. Sending a finite handle across threads in user code is a logic
/// bug; the destination thread's arena does not contain the indexed
/// value and a debug-mode access will panic out of bounds. Sentinel
/// handles, by contrast, carry no arena reference and are
/// thread-independent.
#[derive(Copy, Clone)]
pub struct RationalReal(pub(crate) u32);

// SAFETY: enforced by the per-thread-arena invariant. See module docs.
unsafe impl Send for RationalReal {}
// SAFETY: same.
unsafe impl Sync for RationalReal {}

impl RationalReal {
    /// Construct from an owned `BigRational`. The value is moved into
    /// the thread-local arena and a finite-tagged handle is returned.
    pub fn from_bigrational(value: BigRational) -> Self {
        RationalReal(arena::push(value))
    }

    /// Construct from a numerator/denominator pair. Panics if `den == 0`.
    pub fn from_pair(numer: BigInt, denom: BigInt) -> Self {
        Self::from_bigrational(BigRational::new(numer, denom))
    }

    // ---------- internal tag helpers ----------

    #[inline]
    pub(crate) fn is_finite_tag(self) -> bool {
        arena::is_finite_handle(self.0)
    }

    #[inline]
    pub(crate) fn is_pos_inf_tag(self) -> bool {
        arena::is_pos_inf_handle(self.0)
    }

    #[inline]
    pub(crate) fn is_neg_inf_tag(self) -> bool {
        arena::is_neg_inf_handle(self.0)
    }

    #[inline]
    pub(crate) fn is_infinite_tag(self) -> bool {
        arena::is_infinite_handle(self.0)
    }

    #[inline]
    pub(crate) fn is_nan_tag(self) -> bool {
        arena::is_nan_handle(self.0)
    }

    /// Sign of the value as a small `i32` (`-1`, `0`, or `+1`) for use
    /// in sentinel-propagation arithmetic. NaN inputs return `0`
    /// because callers should already have short-circuited NaN by the
    /// time they call this; treating NaN as zero keeps the function
    /// total and avoids panicking inside the `with` borrow.
    #[inline]
    pub(crate) fn sign_i32(self) -> i32 {
        if self.is_nan_tag() {
            return 0;
        }
        if self.is_pos_inf_tag() {
            return 1;
        }
        if self.is_neg_inf_tag() {
            return -1;
        }
        arena::with(self.0, |a| {
            if a.is_zero() {
                0
            } else if a.is_negative() {
                -1
            } else {
                1
            }
        })
    }

    /// Clone the underlying `BigRational` out of the arena. Panics if
    /// `self` is one of the non-finite sentinels.
    pub fn to_bigrational(&self) -> BigRational {
        if !self.is_finite_tag() {
            panic!(
                "RationalReal::to_bigrational called on a non-finite sentinel \
                 (tag = {}); guard with RealSentinel::is_finite first",
                sentinel_label(self.0)
            );
        }
        arena::get(self.0)
    }

    /// Consume into the `(numerator, denominator)` pair (owned). Panics
    /// if `self` is a sentinel.
    pub fn into_pair(self) -> (BigInt, BigInt) {
        if !self.is_finite_tag() {
            panic!(
                "RationalReal::into_pair called on a non-finite sentinel \
                 (tag = {}); guard with RealSentinel::is_finite first",
                sentinel_label(self.0)
            );
        }
        let r = arena::get(self.0);
        let (n, d) = r.into_raw();
        (n, d)
    }

    /// Lossy conversion to `f64`. Sentinels map to the corresponding
    /// IEEE values: `+inf` → `f64::INFINITY`, `-inf` →
    /// `f64::NEG_INFINITY`, `NaN` → `f64::NAN`. For finite values,
    /// returns `f64::NAN` if the rational magnitude exceeds `f64::MAX`
    /// or if rounding fails (rare; the `num-rational` impl returns
    /// `None` only in extreme edge cases).
    pub fn to_f64(&self) -> f64 {
        if self.is_pos_inf_tag() {
            return f64::INFINITY;
        }
        if self.is_neg_inf_tag() {
            return f64::NEG_INFINITY;
        }
        if self.is_nan_tag() {
            return f64::NAN;
        }
        use num_traits::ToPrimitive;
        arena::with(self.0, |r| r.to_f64().unwrap_or(f64::NAN))
    }

    /// Read-only access to the numerator. Panics if `self` is a sentinel.
    pub fn numer(&self) -> BigInt {
        if !self.is_finite_tag() {
            panic!(
                "RationalReal::numer called on a non-finite sentinel ({})",
                sentinel_label(self.0)
            );
        }
        arena::with(self.0, |r| r.numer().clone())
    }

    /// Read-only access to the denominator. Panics if `self` is a sentinel.
    pub fn denom(&self) -> BigInt {
        if !self.is_finite_tag() {
            panic!(
                "RationalReal::denom called on a non-finite sentinel ({})",
                sentinel_label(self.0)
            );
        }
        arena::with(self.0, |r| r.denom().clone())
    }

    /// Maximum bit-length across numerator and denominator. Cheap; one
    /// `BigInt::bits()` call each. Useful for the per-iteration
    /// denominator-bit-length log requested by QOU. Returns `0` for
    /// sentinel handles (they have no `BigRational` payload).
    pub fn max_bits(&self) -> u64 {
        if !self.is_finite_tag() {
            return 0;
        }
        arena::with(self.0, |r| r.numer().bits().max(r.denom().bits()))
    }

    /// `(numer_bits, denom_bits)` separately. The
    /// [`BitWidthDiagnostic`](crate::algebra::transcendental::BitWidthDiagnostic)
    /// impl on `RationalReal` forwards here; this method is also useful
    /// for direct callers who want a finer-grained breakdown. Returns
    /// `(0, 0)` for sentinel handles.
    pub fn bit_widths(&self) -> (u64, u64) {
        if !self.is_finite_tag() {
            return (0, 0);
        }
        arena::with(self.0, |r| (r.numer().bits(), r.denom().bits()))
    }
}

/// Human-readable name for the non-finite sentinels, used in panic
/// messages to make handle-tag misuse self-explanatory.
fn sentinel_label(handle: u32) -> &'static str {
    if arena::is_pos_inf_handle(handle) {
        "+infinity"
    } else if arena::is_neg_inf_handle(handle) {
        "-infinity"
    } else if arena::is_nan_handle(handle) {
        "NaN"
    } else {
        "finite"
    }
}

impl crate::algebra::transcendental::BitWidthDiagnostic for RationalReal {
    /// Returns `(numer_bits, denom_bits)` of the underlying
    /// `BigRational`. Cheap (one `BigInt::bits()` call each).
    #[inline]
    fn bit_width(&self) -> (u64, u64) {
        self.bit_widths()
    }
}

// ============================================================
// Arithmetic — sentinel-aware. Every op first dispatches on the
// finite/+inf/-inf/NaN tags of its inputs (NaN absorbs, signed
// infinities propagate per IEEE-754 rules), and only when both
// operands are finite does it touch the arena.
// ============================================================

impl Add for RationalReal {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        // NaN + anything = NaN
        if self.is_nan_tag() || rhs.is_nan_tag() {
            return Self::nan_const();
        }
        // ±inf + ∓inf = NaN; otherwise an inf operand absorbs the result.
        if self.is_infinite_tag() || rhs.is_infinite_tag() {
            return match (
                self.is_pos_inf_tag(),
                self.is_neg_inf_tag(),
                rhs.is_pos_inf_tag(),
                rhs.is_neg_inf_tag(),
            ) {
                (true, _, _, true) | (_, true, true, _) => Self::nan_const(),
                (true, _, _, _) | (_, _, true, _) => Self::pos_inf_const(),
                _ => Self::neg_inf_const(),
            };
        }
        // both finite
        let v = arena::with2(self.0, rhs.0, |a, b| a + b);
        RationalReal::from_bigrational(maybe_cap(v))
    }
}

impl Sub for RationalReal {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        // a - b ≡ a + (-b); reuse Add after negating rhs's sentinel tag
        // when needed. Implementing inline keeps NaN propagation cheap.
        if self.is_nan_tag() || rhs.is_nan_tag() {
            return Self::nan_const();
        }
        if self.is_infinite_tag() || rhs.is_infinite_tag() {
            // +inf - +inf or -inf - -inf = NaN
            if (self.is_pos_inf_tag() && rhs.is_pos_inf_tag())
                || (self.is_neg_inf_tag() && rhs.is_neg_inf_tag())
            {
                return Self::nan_const();
            }
            // sign of result determined by whichever side's infinity dominates
            if self.is_pos_inf_tag() || rhs.is_neg_inf_tag() {
                return Self::pos_inf_const();
            }
            return Self::neg_inf_const();
        }
        let v = arena::with2(self.0, rhs.0, |a, b| a - b);
        RationalReal::from_bigrational(maybe_cap(v))
    }
}

impl Mul for RationalReal {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        if self.is_nan_tag() || rhs.is_nan_tag() {
            return Self::nan_const();
        }
        if self.is_infinite_tag() || rhs.is_infinite_tag() {
            // 0 × inf = NaN per IEEE-754
            let s_a = self.sign_i32();
            let s_b = rhs.sign_i32();
            if s_a == 0 || s_b == 0 {
                return Self::nan_const();
            }
            return if s_a * s_b > 0 {
                Self::pos_inf_const()
            } else {
                Self::neg_inf_const()
            };
        }
        let v = arena::with2(self.0, rhs.0, |a, b| a * b);
        RationalReal::from_bigrational(maybe_cap(v))
    }
}

impl Div for RationalReal {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        if self.is_nan_tag() || rhs.is_nan_tag() {
            return Self::nan_const();
        }
        // inf / inf = NaN
        if self.is_infinite_tag() && rhs.is_infinite_tag() {
            return Self::nan_const();
        }
        // finite / inf = +0  (no signed zero in rationals; the magnitude
        // is what matters for the solver and IEEE -0 has no meaningful
        // analogue here).
        if rhs.is_infinite_tag() {
            return Self::zero_arena();
        }
        // inf / finite: sign of inf × sign of rhs (treating ±0 as +).
        if self.is_infinite_tag() {
            let s_a = self.sign_i32(); // ±1
            let s_b = rhs.sign_i32();  // -1 / 0 / +1
            // sign(0) treated as + so inf / +0 = inf with sign of inf.
            let s_b_eff = if s_b == 0 { 1 } else { s_b };
            return if s_a * s_b_eff > 0 {
                Self::pos_inf_const()
            } else {
                Self::neg_inf_const()
            };
        }
        // both finite
        if rhs.is_zero_finite() {
            // x / 0: NaN if x == 0 (0/0), else ±inf with sign of x.
            return if self.is_zero_finite() {
                Self::nan_const()
            } else if self.sign_i32() > 0 {
                Self::pos_inf_const()
            } else {
                Self::neg_inf_const()
            };
        }
        let v = arena::with2(self.0, rhs.0, |a, b| a / b);
        RationalReal::from_bigrational(maybe_cap(v))
    }
}

impl Rem for RationalReal {
    type Output = Self;
    fn rem(self, rhs: Self) -> Self {
        // IEEE-754 fmod-like rules: NaN propagates; inf % anything = NaN;
        // x % 0 = NaN; x % inf = x.
        if self.is_nan_tag() || rhs.is_nan_tag() || self.is_infinite_tag() {
            return Self::nan_const();
        }
        if rhs.is_infinite_tag() {
            return self;
        }
        if rhs.is_zero_finite() {
            return Self::nan_const();
        }
        let v = arena::with2(self.0, rhs.0, |a, b| a % b);
        RationalReal::from_bigrational(maybe_cap(v))
    }
}

impl Neg for RationalReal {
    type Output = Self;
    fn neg(self) -> Self {
        if self.is_pos_inf_tag() {
            return Self::neg_inf_const();
        }
        if self.is_neg_inf_tag() {
            return Self::pos_inf_const();
        }
        if self.is_nan_tag() {
            return Self::nan_const();
        }
        // Neg never grows numerator/denominator bits, so no cap needed.
        let v = arena::with(self.0, |a| -a);
        RationalReal::from_bigrational(v)
    }
}

// ---------- internal sentinel-construction shims ----------
// These avoid the cost of going through the public RealSentinel trait
// (which is used both for f64 and for RationalReal). They construct
// the sentinel handles directly from the constants in `arena`.
impl RationalReal {
    #[inline]
    pub(crate) fn pos_inf_const() -> Self {
        RationalReal(arena::POS_INF_HANDLE)
    }
    #[inline]
    pub(crate) fn neg_inf_const() -> Self {
        RationalReal(arena::NEG_INF_HANDLE)
    }
    #[inline]
    pub(crate) fn nan_const() -> Self {
        RationalReal(arena::NAN_HANDLE)
    }
    /// Push a fresh zero into the arena and return the resulting
    /// finite-tagged handle. Internal shim used by arithmetic operators
    /// (e.g. `finite / inf = 0`) when they need a zero result without
    /// going through the `<Self as Zero>::zero()` trait dispatch.
    #[inline]
    pub(crate) fn zero_arena() -> Self {
        RationalReal(arena::push_zero())
    }
    /// True iff `self` is a finite-tagged handle whose value is zero.
    /// Cheaper than [`Zero::is_zero`] in arithmetic dispatch because
    /// it skips the tag dispatch (the caller has already filtered out
    /// sentinels).
    #[inline]
    pub(crate) fn is_zero_finite(self) -> bool {
        debug_assert!(self.is_finite_tag());
        arena::with(self.0, |a| a.is_zero())
    }
}

impl AddAssign for RationalReal {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl SubAssign for RationalReal {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl MulAssign for RationalReal {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl DivAssign for RationalReal {
    #[inline]
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}

impl RemAssign for RationalReal {
    #[inline]
    fn rem_assign(&mut self, rhs: Self) {
        *self = *self % rhs;
    }
}

// ============================================================
// Comparisons — IEEE-style. NaN is unordered (PartialEq returns
// false for any pair involving NaN; PartialOrd returns None);
// -inf < every finite < +inf. We deliberately do not implement
// Eq or Ord because NaN ≠ NaN breaks Eq's reflexivity contract
// and there is no sensible total order on a type with NaN.
// ============================================================

impl PartialEq for RationalReal {
    fn eq(&self, other: &Self) -> bool {
        // NaN is never equal to anything, including itself (IEEE-754).
        if self.is_nan_tag() || other.is_nan_tag() {
            return false;
        }
        if self.0 == other.0 {
            return true; // same handle, same value (cheap fast path)
        }
        // sentinels of different kinds can never be equal (the same-handle
        // fast path already caught matching sentinels).
        if !self.is_finite_tag() || !other.is_finite_tag() {
            return false;
        }
        arena::with2(self.0, other.0, |a, b| a == b)
    }
}

impl PartialOrd for RationalReal {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        // NaN is unordered with everything, including itself.
        if self.is_nan_tag() || other.is_nan_tag() {
            return None;
        }
        if self.0 == other.0 {
            return Some(Ordering::Equal);
        }
        // -inf is less than every finite and less than +inf.
        if self.is_neg_inf_tag() {
            return Some(Ordering::Less);
        }
        if other.is_neg_inf_tag() {
            return Some(Ordering::Greater);
        }
        // +inf is greater than every finite (and we already returned for
        // self == other, so equal +inf is handled).
        if self.is_pos_inf_tag() {
            return Some(Ordering::Greater);
        }
        if other.is_pos_inf_tag() {
            return Some(Ordering::Less);
        }
        // both finite
        Some(arena::with2(self.0, other.0, |a, b| a.cmp(b)))
    }
}

// ============================================================
// num_traits::Zero / One — required by Num
// ============================================================

impl Zero for RationalReal {
    #[inline]
    fn zero() -> Self {
        RationalReal(arena::push_zero())
    }
    #[inline]
    fn is_zero(&self) -> bool {
        // Sentinels are never zero. Avoids an arena dispatch on the
        // sentinel handle (whose index field has no valid arena entry).
        if !self.is_finite_tag() {
            return false;
        }
        arena::with(self.0, |a| a.is_zero())
    }
}

impl One for RationalReal {
    #[inline]
    fn one() -> Self {
        RationalReal(arena::push_one())
    }
}

// ============================================================
// Default — required by CoreFloatT
// ============================================================

impl Default for RationalReal {
    #[inline]
    fn default() -> Self {
        Self::zero()
    }
}

// ============================================================
// num_traits::Num — required by CoreFloatT (via Num + NumAssign)
// ============================================================

/// Error returned by [`RationalReal::from_str_radix`] when the input
/// cannot be parsed as a rational at the given radix.
#[derive(Debug, Clone)]
pub struct ParseRationalError(String);

impl std::fmt::Display for ParseRationalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "RationalReal parse error: {}", self.0)
    }
}

impl std::error::Error for ParseRationalError {}

impl Num for RationalReal {
    type FromStrRadixErr = ParseRationalError;

    /// Parse `"<numer>/<denom>"` or just `"<numer>"`. Both fields go
    /// through `BigInt::from_str_radix`.
    fn from_str_radix(s: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        let (n, d) = match s.split_once('/') {
            Some((ns, ds)) => (ns, ds),
            None => (s, "1"),
        };
        let numer = BigInt::parse_bytes(n.as_bytes(), radix)
            .ok_or_else(|| ParseRationalError(format!("invalid numerator: {n}")))?;
        let denom = BigInt::parse_bytes(d.as_bytes(), radix)
            .ok_or_else(|| ParseRationalError(format!("invalid denominator: {d}")))?;
        if denom.is_zero() {
            return Err(ParseRationalError("zero denominator".into()));
        }
        Ok(Self::from_pair(numer, denom))
    }
}

// ============================================================
// num_traits::Signed — abs / signum / is_positive / is_negative
// ============================================================

impl Signed for RationalReal {
    fn abs(&self) -> Self {
        if self.is_nan_tag() {
            return Self::nan_const();
        }
        if self.is_infinite_tag() {
            // |±inf| = +inf
            return Self::pos_inf_const();
        }
        let v = arena::with(self.0, |a| a.abs());
        RationalReal::from_bigrational(v)
    }

    fn abs_sub(&self, other: &Self) -> Self {
        // |x - y| if x > y, else 0. NaN-poisoned per IEEE.
        if self.is_nan_tag() || other.is_nan_tag() {
            return Self::nan_const();
        }
        if self <= other {
            Self::zero()
        } else {
            *self - *other
        }
    }

    fn signum(&self) -> Self {
        // sign(NaN) = NaN; sign(±inf) = ±1.
        if self.is_nan_tag() {
            return Self::nan_const();
        }
        if self.is_pos_inf_tag() {
            return Self::from_pair(BigInt::from(1), BigInt::from(1));
        }
        if self.is_neg_inf_tag() {
            return Self::from_pair(BigInt::from(-1), BigInt::from(1));
        }
        let v = arena::with(self.0, |a| a.signum());
        RationalReal::from_bigrational(v)
    }

    fn is_positive(&self) -> bool {
        if self.is_pos_inf_tag() {
            return true;
        }
        if self.is_neg_inf_tag() || self.is_nan_tag() {
            return false;
        }
        arena::with(self.0, |a| a.is_positive())
    }

    fn is_negative(&self) -> bool {
        if self.is_neg_inf_tag() {
            return true;
        }
        if self.is_pos_inf_tag() || self.is_nan_tag() {
            return false;
        }
        arena::with(self.0, |a| a.is_negative())
    }
}

// ============================================================
// num_traits::FromPrimitive — used by AsFloatT (T::from_usize etc)
// ============================================================

impl FromPrimitive for RationalReal {
    fn from_i64(n: i64) -> Option<Self> {
        Some(Self::from_pair(BigInt::from(n), BigInt::from(1)))
    }
    fn from_u64(n: u64) -> Option<Self> {
        Some(Self::from_pair(BigInt::from(n), BigInt::from(1)))
    }
    fn from_i128(n: i128) -> Option<Self> {
        Some(Self::from_pair(BigInt::from(n), BigInt::from(1)))
    }
    fn from_u128(n: u128) -> Option<Self> {
        Some(Self::from_pair(BigInt::from(n), BigInt::from(1)))
    }
    fn from_isize(n: isize) -> Option<Self> {
        Self::from_i64(n as i64)
    }
    fn from_usize(n: usize) -> Option<Self> {
        Self::from_u64(n as u64)
    }
    fn from_i32(n: i32) -> Option<Self> {
        Self::from_i64(n as i64)
    }
    fn from_u32(n: u32) -> Option<Self> {
        Self::from_u64(n as u64)
    }
    /// `f32` is converted exactly into `(mantissa * 2^exp_bias)` form so
    /// that, e.g., `from_f32(0.1)` is the *exact* rational the IEEE
    /// representation 0x3DCCCCCD denotes (not the round-trip back to 1/10).
    /// Non-finite inputs (`NaN`, `±Inf`) map to the corresponding
    /// sentinel handles.
    fn from_f32(f: f32) -> Option<Self> {
        if f.is_nan() {
            return Some(Self::nan_const());
        }
        if f == f32::INFINITY {
            return Some(Self::pos_inf_const());
        }
        if f == f32::NEG_INFINITY {
            return Some(Self::neg_inf_const());
        }
        BigRational::from_f32(f).map(Self::from_bigrational)
    }
    /// `f64` is converted exactly. See [`Self::from_f32`].
    fn from_f64(f: f64) -> Option<Self> {
        if f.is_nan() {
            return Some(Self::nan_const());
        }
        if f == f64::INFINITY {
            return Some(Self::pos_inf_const());
        }
        if f == f64::NEG_INFINITY {
            return Some(Self::neg_inf_const());
        }
        BigRational::from_f64(f).map(Self::from_bigrational)
    }
}

// ============================================================
// From conversions for ergonomics
// ============================================================

impl From<BigRational> for RationalReal {
    fn from(v: BigRational) -> Self {
        Self::from_bigrational(v)
    }
}

impl From<i64> for RationalReal {
    fn from(n: i64) -> Self {
        Self::from_pair(BigInt::from(n), BigInt::from(1))
    }
}

impl From<f64> for RationalReal {
    fn from(f: f64) -> Self {
        // Non-finite inputs map to the corresponding sentinel handles
        // (matches the FromPrimitive::from_f64 behaviour added by the
        // sentinel bit-tagging redesign).
        if f.is_nan() {
            return Self::nan_const();
        }
        if f == f64::INFINITY {
            return Self::pos_inf_const();
        }
        if f == f64::NEG_INFINITY {
            return Self::neg_inf_const();
        }
        Self::from_f64(f).unwrap_or_else(|| panic!("cannot convert f64 ({f}) to RationalReal"))
    }
}
