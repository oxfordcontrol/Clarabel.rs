//! GMP/MPFR interop for [`RationalReal`].
//!
//! Available when the `rug-interop` cargo feature is enabled. Provides
//! bit-exact conversion from [`rug::Rational`] (a thin GMP `mpq_t`
//! wrapper) into `RationalReal`. Useful for downstream crates that
//! carry MPFR rationals (e.g. QOU's `seminormal_mpfr.rs`) and want
//! to feed them into the solver without going through `f64`.
//!
//! The conversion is bit-exact: rug's numerator/denominator are
//! formatted as base-10 strings (via the GMP `mpz_get_str` path),
//! parsed back as `num_bigint::BigInt`, and assembled into a
//! `BigRational`. No precision is lost in either direction.

use super::real::RationalReal;
use num_bigint::BigInt;
use num_rational::BigRational;

impl From<&rug::Rational> for RationalReal {
    /// Bit-exact conversion of an MPFR/GMP rational into a
    /// `RationalReal`. Numerator and denominator round-trip through
    /// their base-10 string representations (GMP guarantees this is
    /// lossless for arbitrary precision).
    fn from(r: &rug::Rational) -> Self {
        // rug::Integer's Display impl writes base-10 unconditionally;
        // round-trip via that to BigInt is lossless.
        let n_str = r.numer().to_string();
        let d_str = r.denom().to_string();
        let numer = BigInt::parse_bytes(n_str.as_bytes(), 10)
            .expect("rug numerator must parse as BigInt");
        let denom = BigInt::parse_bytes(d_str.as_bytes(), 10)
            .expect("rug denominator must parse as BigInt");
        RationalReal::from_bigrational(BigRational::new(numer, denom))
    }
}

impl From<rug::Rational> for RationalReal {
    fn from(r: rug::Rational) -> Self {
        Self::from(&r)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::rational::reset_arena;

    #[test]
    fn rug_rational_round_trips_bit_exactly() {
        reset_arena();
        // 22/7 in rug -> RationalReal -> into_pair must give exactly
        // (22, 7).
        let r = rug::Rational::from((22, 7));
        let rr: RationalReal = (&r).into();
        let (n, d) = rr.into_pair();
        assert_eq!(n.to_string(), "22");
        assert_eq!(d.to_string(), "7");
        reset_arena();
    }

    #[test]
    fn rug_large_numerator_preserved() {
        reset_arena();
        // 2^200 / 3 — beyond any f64 magnitude. Must survive
        // intact through the string round-trip.
        let big_n_str = {
            let mut s = String::from("1");
            for _ in 0..60 {
                s.push('0');
            }
            s
        };
        let r = rug::Rational::from_str_radix(&format!("{}/3", big_n_str), 10).unwrap();
        let rr: RationalReal = (&r).into();
        let (n, d) = rr.into_pair();
        assert_eq!(n.to_string(), big_n_str);
        assert_eq!(d.to_string(), "3");
        reset_arena();
    }
}
