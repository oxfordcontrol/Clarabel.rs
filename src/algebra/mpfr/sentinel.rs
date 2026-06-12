use super::arena;
use super::real::MpfrFloat;
use crate::algebra::transcendental::{RealConst, RealSentinel};
use rug::Float as RugFloat;
use rug::float::Special;

impl RealSentinel for MpfrFloat {
    fn infinity() -> Self { MpfrFloat(arena::push(RugFloat::with_val(super::default_precision(), Special::Infinity))) }
    fn neg_infinity() -> Self { MpfrFloat(arena::push(RugFloat::with_val(super::default_precision(), Special::NegInfinity))) }
    fn nan() -> Self { MpfrFloat(arena::push(RugFloat::with_val(super::default_precision(), Special::Nan))) }
    fn epsilon() -> Self {
        let p = super::default_precision();
        let mut e = RugFloat::with_val(p, - (p as i32));
        e.exp2_mut(); // e = 2^-p
        MpfrFloat(arena::push(e))
    }
    fn is_nan(self) -> bool { arena::with(self.0, |a| a.is_nan()) }
    fn is_infinite(self) -> bool { arena::with(self.0, |a| a.is_infinite()) }
    fn is_finite(self) -> bool { arena::with(self.0, |a| a.is_finite()) }
    fn max_value() -> Self { Self::infinity() }
    fn min_value() -> Self { Self::neg_infinity() }
    fn is_sign_negative(self) -> bool { arena::with(self.0, |a| a.is_sign_negative()) }
    fn min(self, other: Self) -> Self { if self < other { self } else { other } }
    fn max(self, other: Self) -> Self { if self > other { self } else { other } }
}

impl RealConst for MpfrFloat {
    fn FRAC_1_SQRT_2() -> Self {
        let p = super::default_precision();
        let val = RugFloat::with_val(p, 2).sqrt().recip();
        MpfrFloat(arena::push(val))
    }
    fn PI() -> Self {
        let p = super::default_precision();
        let val = RugFloat::with_val(p, rug::float::Constant::Pi);
        MpfrFloat(arena::push(val))
    }
    fn SQRT_2() -> Self {
        let p = super::default_precision();
        let val = RugFloat::with_val(p, 2).sqrt();
        MpfrFloat(arena::push(val))
    }
}
