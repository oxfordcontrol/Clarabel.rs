//! Unit tests for the `RationalReal` arena backend.

#![allow(unused_imports)]

use super::*;
use crate::algebra::transcendental::{RealConst, RealSentinel, Transcendental};
use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{FromPrimitive, One, Signed, Zero};

// Bring `arena::get` into scope for the cache-invalidation test.
use super::arena;

#[test]
fn rational_arena_smoke() {
    reset_arena();
    let _a: RationalReal = 1i64.into();
    let _b: RationalReal = 2i64.into();
    assert!(arena_len() >= 2);
    reset_arena();
    assert_eq!(arena_len(), 0);
}

#[test]
fn rational_basic_arithmetic_is_exact() {
    reset_arena();
    let third: RationalReal =
        RationalReal::from_pair(BigInt::from(1), BigInt::from(3));
    let three_thirds = third + third + third;
    let one = RationalReal::one();
    assert!(
        three_thirds == one,
        "1/3 + 1/3 + 1/3 should be exactly 1, got {:?}",
        three_thirds
    );
    reset_arena();
}

#[test]
fn rational_into_pair_round_trips() {
    reset_arena();
    let r: RationalReal =
        RationalReal::from_pair(BigInt::from(22), BigInt::from(7));
    let (n, d) = r.into_pair();
    assert_eq!(n, BigInt::from(22));
    assert_eq!(d, BigInt::from(7));
    reset_arena();
}

#[test]
fn rational_from_f64_is_exact_in_binary() {
    // 0.1 has no exact binary representation; from_f64 captures the
    // *exact* IEEE bit pattern rather than the decimal "1/10".
    reset_arena();
    let r = RationalReal::from(0.1f64);
    assert!(
        (r.to_f64() - 0.1f64).abs() < f64::EPSILON,
        "round-trip 0.1 -> RationalReal -> f64 within ulp"
    );
    reset_arena();
}

#[test]
fn rational_powi_is_exact() {
    reset_arena();
    let half = RationalReal::from_pair(BigInt::from(1), BigInt::from(2));
    let one_over_eight = half.powi(3);
    let expected = RationalReal::from_pair(BigInt::from(1), BigInt::from(8));
    assert_eq!(one_over_eight, expected);
    reset_arena();
}

#[test]
fn rational_sqrt_two_squared_close_to_two_at_128_bits() {
    reset_arena();
    set_precision_bits(128);
    let two_r = RationalReal::from_i64(2).unwrap();
    let s = two_r.sqrt();
    let back = s * s;
    let diff = (back - two_r).abs();
    let tol = RationalReal::from_pair(BigInt::one(), BigInt::one() << 100);
    assert!(diff < tol, "sqrt(2)² should match 2 to within 2^-100");
    reset_arena();
}

#[test]
fn rational_exp_ln_round_trip() {
    reset_arena();
    set_precision_bits(128);
    for v in [0.5_f64, 1.0_f64, 2.0_f64, 100.0_f64] {
        let x = RationalReal::from(v);
        let back = x.ln().exp();
        let err = (back - x).abs();
        let tol = RationalReal::from_pair(BigInt::one(), BigInt::one() << 80);
        assert!(
            err < tol,
            "exp(ln({v})) should match {v} to within 2^-80, got err = {err:?}"
        );
    }
    reset_arena();
}

#[test]
fn rational_is_floatt() {
    // The compile-time assertion in mod.rs already proves this; here we
    // also verify a few trait method calls work at run time.
    reset_arena();
    let inf = <RationalReal as RealSentinel>::infinity();
    assert!(inf.is_infinite());
    assert!(!inf.is_finite());

    let pi = <RationalReal as RealConst>::PI();
    assert!((pi.to_f64() - std::f64::consts::PI).abs() < 1e-12);
    reset_arena();
}

#[test]
fn rational_recip_of_zero_is_infinity_not_panic() {
    // IEEE 1.0/0.0 = +infinity. We match that semantics via the
    // RealSentinel infinity, rather than panicking on BigRational
    // divide-by-zero. Solver hot paths (KKT diagonal, equilibration)
    // depend on this not aborting.
    reset_arena();
    let z = RationalReal::zero();
    let r = z.recip();
    assert!(<RationalReal as RealSentinel>::is_infinite(r));
    reset_arena();
}

#[test]
fn rational_powf_zero_base_handles_edge_cases() {
    // 0^positive = 0; 0^0 = 1; 0^negative = nan.
    reset_arena();
    let z = RationalReal::zero();
    let pos = RationalReal::from_i64(2).unwrap();
    let neg = RationalReal::from_i64(-2).unwrap();
    let zero_to_pos = z.powf(pos);
    assert!(zero_to_pos.is_zero());
    let zero_to_zero = z.powf(z);
    assert_eq!(zero_to_zero, RationalReal::one());
    let zero_to_neg = z.powf(neg);
    assert!(<RationalReal as RealSentinel>::is_nan(zero_to_neg));
    reset_arena();
}

#[test]
fn rational_pi_high_precision_matches_taylor_at_300_bits() {
    // PI at 300 bits should agree with std::f64::consts::PI to far
    // beyond f64 ulp (the ulp is ~2^-52, the PI evaluation at 300
    // bits is good to ~2^-300).
    reset_arena();
    set_precision_bits(300);
    let pi = <RationalReal as RealConst>::PI();
    let pi_f = pi.to_f64();
    let err = (pi_f - std::f64::consts::PI).abs();
    assert!(err < 1e-15, "PI at 300 bits matches f64::PI within ulp; err = {err}");

    // Also: 4 * pi should equal 4*PI (sanity for the const cache /
    // arena handle path).
    let four_pi = <RationalReal as RationalReal_helper>::quadruple(pi);
    assert!(
        (four_pi.to_f64() - 4.0 * std::f64::consts::PI).abs() < 1e-14,
        "4*PI sanity"
    );
    set_precision_bits(128);
    reset_arena();
}

trait RationalReal_helper: Sized {
    fn quadruple(self) -> Self;
}
impl RationalReal_helper for RationalReal {
    fn quadruple(self) -> Self {
        self + self + self + self
    }
}

#[test]
fn rational_sqrt_2_high_precision_squared_close_to_2() {
    // SQRT_2 at 200 bits, squared, should match 2 within 2^-150.
    reset_arena();
    set_precision_bits(200);
    let s = <RationalReal as RealConst>::SQRT_2();
    let back = s * s;
    let two = RationalReal::from_i64(2).unwrap();
    let diff = (back - two).abs();
    let tol = RationalReal::from_pair(BigInt::one(), BigInt::one() << 150);
    assert!(diff < tol, "SQRT_2² should match 2 to within 2^-150 at 200-bit precision");
    set_precision_bits(128);
    reset_arena();
}

#[test]
fn rational_max_bits_grows_with_repeated_division() {
    reset_arena();
    let mut x = RationalReal::from_i64(1).unwrap();
    let three = RationalReal::from_i64(3).unwrap();
    let initial_bits = x.max_bits();
    for _ in 0..10 {
        x = x / three;
    }
    let after_bits = x.max_bits();
    assert!(
        after_bits > initial_bits,
        "denominator should grow as 3^10 ≈ 16 bits over 10 divisions"
    );
    reset_arena();
}

#[test]
fn rational_max_arena_bits_caps_growth() {
    // With cap on at 256 bits, repeated 1/3 divisions stay bounded
    // (3^-50 needs only ~80 denom bits, so 256 leaves plenty of margin
    // and the capped value matches the true 3^-50 to ulp in f64).
    reset_arena();
    set_max_arena_bits(Some(256));
    let mut x = RationalReal::from_i64(1).unwrap();
    let three = RationalReal::from_i64(3).unwrap();
    for _ in 0..50 {
        x = x / three;
    }
    let bits = x.max_bits();
    assert!(
        bits <= 256,
        "max_arena_bits(256) should bound denom+numer; got {bits} bits after 50 divisions"
    );
    // Result is approximately 3^-50; check the f64 view.
    let approx = x.to_f64();
    let expected = 3f64.powi(-50);
    let rel_err = (approx - expected).abs() / expected.abs();
    assert!(
        rel_err < 1e-15,
        "capped result {approx} should match 3^-50 = {expected} to ~ulp; rel_err = {rel_err}"
    );
    set_max_arena_bits(None);
    reset_arena();
}

#[test]
fn rational_aggressive_cap_rounds_to_zero_when_value_smaller_than_ulp() {
    // Documents the boundary: at 32-bit cap, anything smaller than
    // ~2^-32 rounds to exactly 0. This is intentional — the cap is
    // a precision floor, not a magnitude floor.
    reset_arena();
    set_max_arena_bits(Some(32));
    let small = RationalReal::from_pair(BigInt::from(1), BigInt::from(1u64) << 50);
    // small = 2^-50, far below 2^-32 ulp at 32-bit precision -> rounds to 0.
    let bumped = small + RationalReal::zero(); // any op triggers cap
    assert_eq!(bumped.to_f64(), 0.0);
    set_max_arena_bits(None);
    reset_arena();
}

#[test]
fn rational_with_max_arena_bits_scope_guard_restores_on_panic() {
    reset_arena();
    let original = max_arena_bits();
    let _ = std::panic::catch_unwind(|| {
        with_max_arena_bits(Some(32), || {
            assert_eq!(max_arena_bits(), Some(32));
            panic!("oops");
        });
    });
    assert_eq!(
        max_arena_bits(),
        original,
        "with_max_arena_bits guard must restore the previous value even on panic"
    );
    reset_arena();
}

#[test]
fn rational_sentinel_handle_is_arena_independent() {
    // After the bit-tagging redesign, sentinels are encoded purely in
    // the top two bits of the u32 handle and consume no arena slots.
    // The handle is a compile-time constant: it is identical before
    // and after reset_arena(), and unaffected by intervening pushes.
    reset_arena();
    let inf1 = <RationalReal as RealSentinel>::infinity();
    assert!(inf1.is_infinite() && !inf1.is_finite());

    // Push a bunch of unrelated values to grow the arena, reset, push
    // more, and re-fetch infinity. The bit pattern must be identical
    // and the predicate must still hold.
    for _ in 0..10 {
        let _ = RationalReal::from_i64(42).unwrap();
    }
    reset_arena();
    let a = RationalReal::from_i64(7).unwrap();
    let b = RationalReal::from_i64(11).unwrap();
    let _ = a + b;
    let _ = a * b;

    let inf2 = <RationalReal as RealSentinel>::infinity();
    assert!(inf2.is_infinite(), "infinity() after reset must still be infinite");
    assert_eq!(
        inf1.0, inf2.0,
        "sentinel handles are pure tag bits and must be identical across resets"
    );
    reset_arena();
}

#[cfg(feature = "serde")]
#[test]
fn rational_serde_round_trip_preserves_exact_value() {
    use serde_json;
    reset_arena();
    // 22/7 has no exact f64 representation; the round trip must
    // preserve the (numer, denom) pair exactly even after the
    // arena is reset between serialize and deserialize.
    let r = RationalReal::from_pair(BigInt::from(22), BigInt::from(7));
    let s = serde_json::to_string(&r).unwrap();
    reset_arena(); // simulate transport across a process boundary
    let back: RationalReal = serde_json::from_str(&s).unwrap();
    let (n, d) = back.into_pair();
    assert_eq!(n, BigInt::from(22));
    assert_eq!(d, BigInt::from(7));
    reset_arena();
}

#[test]
fn rational_one_third_plus_one_third_plus_one_third_still_one_in_exact_mode() {
    // Default mode (no cap) preserves the headline guarantee that
    // 1/3 + 1/3 + 1/3 == 1 exactly.
    reset_arena();
    set_max_arena_bits(None); // explicit, even though it's the default
    let third = RationalReal::from_pair(BigInt::from(1), BigInt::from(3));
    let three_thirds = third + third + third;
    assert_eq!(three_thirds, RationalReal::one());
    reset_arena();
}

// ============================================================
// Sentinel bit-tagging propagation tests.
//
// These verify the three correctness issues flagged by Gemini in the
// PR #1 review of the original handle-equality sentinel design:
//
// 1. Range — no finite arena value can ever exceed +infinity or be
//    below -infinity, regardless of magnitude.
// 2. Propagation — sentinel inputs survive arithmetic instead of
//    devolving back into finite handles.
// 3. Predicates — is_finite / is_infinite / is_nan operate on the
//    handle's tag bits, so they correctly classify any value that
//    arose from a sentinel-touching operation.
// ============================================================

#[test]
fn rational_sentinel_range_no_finite_value_exceeds_infinity() {
    // Construct a finite value of magnitude 2^512, far above the
    // 2^256 fudge that the previous design used as the +inf magnitude.
    // The new tag-bit encoding makes this a nonissue: any finite,
    // however large, sorts strictly below +infinity and strictly
    // above -infinity.
    reset_arena();
    let big_pos = RationalReal::from_pair(BigInt::one() << 512u32, BigInt::one());
    let big_neg = RationalReal::from_pair(-(BigInt::one() << 512u32), BigInt::one());
    let inf = <RationalReal as RealSentinel>::infinity();
    let neg_inf = <RationalReal as RealSentinel>::neg_infinity();
    assert!(big_pos < inf, "2^512 must be < +inf");
    assert!(big_neg > neg_inf, "-2^512 must be > -inf");
    assert!(big_pos.is_finite());
    assert!(big_neg.is_finite());
    reset_arena();
}

#[test]
fn rational_sentinel_propagates_through_addition() {
    reset_arena();
    let inf = <RationalReal as RealSentinel>::infinity();
    let one = RationalReal::one();
    let r = inf + one;
    assert!(r.is_infinite() && !r.is_finite(), "inf + 1 must be infinite");
    assert!(!r.is_sign_negative(), "inf + 1 must be positive");
    let neg_inf = <RationalReal as RealSentinel>::neg_infinity();
    let r2 = inf + neg_inf;
    assert!(r2.is_nan(), "inf + (-inf) must be NaN");
    reset_arena();
}

#[test]
fn rational_sentinel_propagates_through_multiplication() {
    reset_arena();
    let inf = <RationalReal as RealSentinel>::infinity();
    let two = RationalReal::from_i64(2).unwrap();
    let neg_two = RationalReal::from_i64(-2).unwrap();
    let zero = RationalReal::zero();
    assert!((inf * two).is_infinite() && !(inf * two).is_sign_negative());
    assert!((inf * neg_two).is_infinite() && (inf * neg_two).is_sign_negative());
    assert!((inf * zero).is_nan(), "inf * 0 must be NaN");
    reset_arena();
}

#[test]
fn rational_sentinel_propagates_through_division() {
    reset_arena();
    let inf = <RationalReal as RealSentinel>::infinity();
    let two = RationalReal::from_i64(2).unwrap();
    let zero = RationalReal::zero();
    // inf / inf = NaN
    assert!((inf / inf).is_nan());
    // finite / inf = 0
    let r = two / inf;
    assert!(r.is_zero(), "2 / inf must be 0");
    // 0 / 0 = NaN
    assert!((zero / zero).is_nan());
    // 1 / 0 = +inf
    let one = RationalReal::one();
    let r2 = one / zero;
    assert!(r2.is_infinite() && !r2.is_sign_negative());
    reset_arena();
}

#[test]
fn rational_sentinel_negation_flips_infinity_sign() {
    reset_arena();
    let inf = <RationalReal as RealSentinel>::infinity();
    let neg_inf = <RationalReal as RealSentinel>::neg_infinity();
    let nan = <RationalReal as RealSentinel>::nan();
    assert_eq!((-inf).0, neg_inf.0);
    assert_eq!((-neg_inf).0, inf.0);
    // -NaN is NaN (bit-tag-only; no IEEE-style sign-bit retention)
    assert!((-nan).is_nan());
    reset_arena();
}

#[test]
fn rational_sentinel_predicates_track_value_through_arithmetic() {
    // The crux of Gemini's predicate complaint: previously is_finite
    // was a handle-equality check, so (inf + 1) was a fresh finite
    // handle and is_finite() returned the wrong answer. With tag-bit
    // propagation, the post-arithmetic handle still has the +inf tag
    // and is_finite/is_infinite/is_nan all return the correct value.
    reset_arena();
    let inf = <RationalReal as RealSentinel>::infinity();
    let derived = inf + RationalReal::one();
    assert!(derived.is_infinite());
    assert!(!derived.is_finite());
    assert!(!derived.is_nan());
    let nan_derived = (inf - inf) * RationalReal::from_i64(7).unwrap();
    assert!(nan_derived.is_nan());
    assert!(!nan_derived.is_finite());
    assert!(!nan_derived.is_infinite());
    reset_arena();
}

#[test]
fn rational_sentinel_nan_is_unordered_and_unequal() {
    reset_arena();
    let nan = <RationalReal as RealSentinel>::nan();
    let one = RationalReal::one();
    // PartialEq: NaN != NaN, NaN != 1
    assert!(nan != nan);
    assert!(nan != one);
    assert!(one != nan);
    // PartialOrd: NaN comparisons return None; <, >, <= and >= all false.
    assert!(nan.partial_cmp(&one).is_none());
    assert!(nan.partial_cmp(&nan).is_none());
    assert!(!(nan < one));
    assert!(!(nan > one));
    assert!(!(nan <= one));
    assert!(!(nan >= one));
    reset_arena();
}

#[test]
fn rational_sentinel_min_max_are_nan_poisoned() {
    reset_arena();
    let nan = <RationalReal as RealSentinel>::nan();
    let one = RationalReal::one();
    let inf = <RationalReal as RealSentinel>::infinity();
    let neg_inf = <RationalReal as RealSentinel>::neg_infinity();
    assert!(<RationalReal as RealSentinel>::min(nan, one).is_nan());
    assert!(<RationalReal as RealSentinel>::max(one, nan).is_nan());
    // -inf is the floor; +inf is the ceiling
    assert_eq!(<RationalReal as RealSentinel>::min(neg_inf, one).0, neg_inf.0);
    assert_eq!(<RationalReal as RealSentinel>::max(inf, one).0, inf.0);
    reset_arena();
}

#[test]
fn rational_sentinel_to_f64_maps_to_ieee_specials() {
    reset_arena();
    let inf = <RationalReal as RealSentinel>::infinity();
    let neg_inf = <RationalReal as RealSentinel>::neg_infinity();
    let nan = <RationalReal as RealSentinel>::nan();
    assert_eq!(inf.to_f64(), f64::INFINITY);
    assert_eq!(neg_inf.to_f64(), f64::NEG_INFINITY);
    assert!(nan.to_f64().is_nan());
    reset_arena();
}

#[test]
fn rational_powf_ieee_special_cases() {
    // IEEE-754 / C99 pow special cases that survived the
    // bit-tagging review (Gemini PR #2 review):
    //
    //   pow(x, 0) = 1 for any x, including NaN and ±inf
    //   pow(1, y) = 1 for any y, including NaN and ±inf
    //   pow(-inf, +inf) = +inf
    //   pow(-inf, -inf) = +0
    reset_arena();
    let inf = <RationalReal as RealSentinel>::infinity();
    let neg_inf = <RationalReal as RealSentinel>::neg_infinity();
    let nan = <RationalReal as RealSentinel>::nan();
    let zero = RationalReal::zero();
    let one = RationalReal::one();

    // x^0 = 1 for any x (NaN, ±inf, 0)
    assert_eq!(nan.powf(zero), one, "pow(NaN, 0) must be 1");
    assert_eq!(inf.powf(zero), one, "pow(+inf, 0) must be 1");
    assert_eq!(neg_inf.powf(zero), one, "pow(-inf, 0) must be 1");
    assert_eq!(zero.powf(zero), one, "pow(0, 0) must be 1");

    // 1^y = 1 for any y (NaN, ±inf)
    assert_eq!(one.powf(nan), one, "pow(1, NaN) must be 1");
    assert_eq!(one.powf(inf), one, "pow(1, +inf) must be 1");
    assert_eq!(one.powf(neg_inf), one, "pow(1, -inf) must be 1");

    // pow(-inf, ±inf)
    assert!(neg_inf.powf(inf).is_infinite() && !neg_inf.powf(inf).is_sign_negative(),
        "pow(-inf, +inf) must be +inf");
    assert!(neg_inf.powf(neg_inf).is_zero(),
        "pow(-inf, -inf) must be +0");

    reset_arena();
}

#[test]
fn rational_transcendental_propagates_sentinels() {
    reset_arena();
    let inf = <RationalReal as RealSentinel>::infinity();
    let neg_inf = <RationalReal as RealSentinel>::neg_infinity();
    let nan = <RationalReal as RealSentinel>::nan();
    let zero = RationalReal::zero();
    // sqrt
    assert!(<RationalReal as Transcendental>::sqrt(inf).is_infinite());
    assert!(<RationalReal as Transcendental>::sqrt(neg_inf).is_nan());
    assert!(<RationalReal as Transcendental>::sqrt(nan).is_nan());
    // ln
    assert!(<RationalReal as Transcendental>::ln(inf).is_infinite());
    assert!(<RationalReal as Transcendental>::ln(zero).is_infinite()
        && <RationalReal as Transcendental>::ln(zero).is_sign_negative());
    assert!(<RationalReal as Transcendental>::ln(nan).is_nan());
    // exp
    assert!(<RationalReal as Transcendental>::exp(inf).is_infinite());
    assert!(<RationalReal as Transcendental>::exp(neg_inf).is_zero());
    assert!(<RationalReal as Transcendental>::exp(nan).is_nan());
    // recip
    assert!(<RationalReal as Transcendental>::recip(inf).is_zero());
    assert!(<RationalReal as Transcendental>::recip(zero).is_infinite());
    // powi
    assert!(<RationalReal as Transcendental>::powi(inf, 0) == RationalReal::one());
    assert!(<RationalReal as Transcendental>::powi(inf, 3).is_infinite());
    assert!(<RationalReal as Transcendental>::powi(inf, -2).is_zero());
    reset_arena();
}

#[test]
fn rational_sentinels_consume_no_arena_slots() {
    // Sentinel handles are pure tag bits; constructing them should
    // not push anything into the arena. Contrast the previous design,
    // where ensure_sentinels() pushed three BigRationals on first use.
    reset_arena();
    let baseline = arena_len();
    let _inf = <RationalReal as RealSentinel>::infinity();
    let _neg_inf = <RationalReal as RealSentinel>::neg_infinity();
    let _nan = <RationalReal as RealSentinel>::nan();
    assert_eq!(arena_len(), baseline, "sentinels must not allocate arena slots");
    reset_arena();
}

