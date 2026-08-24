use super::arena;
use super::precision::default_precision;
use num_traits::{FromPrimitive, Num, One, Signed, Zero};
use rug::Float as RugFloat;
use std::cmp::Ordering;
use std::ops::{
    Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign,
};

#[derive(Clone, Copy)]
pub struct MpfrFloat(pub(crate) u32);

unsafe impl Send for MpfrFloat {}
unsafe impl Sync for MpfrFloat {}

impl MpfrFloat {
    pub fn from_rug(f: RugFloat) -> Self {
        MpfrFloat(arena::push(f))
    }

    pub fn as_rug(&self) -> RugFloat {
        arena::get(self.0)
    }

    pub fn into_rug(self) -> RugFloat {
        arena::get(self.0)
    }

    pub fn zero_with_prec(prec: u32) -> Self {
        MpfrFloat(arena::push(RugFloat::new(prec)))
    }

    pub fn with_val<V>(value: V) -> Self
    where
        RugFloat: rug::Assign<V>,
    {
        let mut f = RugFloat::new(default_precision());
        rug::Assign::assign(&mut f, value);
        MpfrFloat(arena::push(f))
    }

    pub fn to_f64(&self) -> f64 {
        arena::with(self.0, |f| f.to_f64())
    }

    pub fn prec(&self) -> u32 {
        arena::with(self.0, |f| f.prec())
    }
}

#[inline]
fn binop_prec(a: &RugFloat, b: &RugFloat) -> u32 {
    a.prec().max(b.prec())
}

impl Add for MpfrFloat {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        let val = arena::with2(self.0, rhs.0, |a, b| {
            let p = binop_prec(a, b);
            RugFloat::with_val(p, a + b)
        });
        MpfrFloat(arena::push(val))
    }
}

impl Sub for MpfrFloat {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        let val = arena::with2(self.0, rhs.0, |a, b| {
            let p = binop_prec(a, b);
            RugFloat::with_val(p, a - b)
        });
        MpfrFloat(arena::push(val))
    }
}

impl Mul for MpfrFloat {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        let val = arena::with2(self.0, rhs.0, |a, b| {
            let p = binop_prec(a, b);
            RugFloat::with_val(p, a * b)
        });
        MpfrFloat(arena::push(val))
    }
}

impl Div for MpfrFloat {
    type Output = Self;
    #[inline]
    fn div(self, rhs: Self) -> Self {
        let val = arena::with2(self.0, rhs.0, |a, b| {
            let p = binop_prec(a, b);
            RugFloat::with_val(p, a / b)
        });
        MpfrFloat(arena::push(val))
    }
}

impl Rem for MpfrFloat {
    type Output = Self;
    #[inline]
    fn rem(self, rhs: Self) -> Self {
        let val = arena::with2(self.0, rhs.0, |a, b| {
            let p = binop_prec(a, b);
            let mut out = RugFloat::with_val(p, a);
            out.remainder_round(b, rug::float::Round::Nearest);
            out
        });
        MpfrFloat(arena::push(val))
    }
}

impl Neg for MpfrFloat {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        let val = arena::with(self.0, |a| {
            let p = a.prec();
            RugFloat::with_val(p, -a)
        });
        MpfrFloat(arena::push(val))
    }
}

impl AddAssign for MpfrFloat {
    #[inline]
    fn add_assign(&mut self, rhs: Self) { *self = *self + rhs; }
}

impl SubAssign for MpfrFloat {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) { *self = *self - rhs; }
}

impl MulAssign for MpfrFloat {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) { *self = *self * rhs; }
}

impl DivAssign for MpfrFloat {
    #[inline]
    fn div_assign(&mut self, rhs: Self) { *self = *self / rhs; }
}

impl RemAssign for MpfrFloat {
    #[inline]
    fn rem_assign(&mut self, rhs: Self) { *self = *self % rhs; }
}

impl PartialEq for MpfrFloat {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        arena::with2(self.0, other.0, |a, b| a == b)
    }
}

impl PartialOrd for MpfrFloat {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        arena::with2(self.0, other.0, |a, b| a.partial_cmp(b))
    }
}

impl Zero for MpfrFloat {
    #[inline]
    fn zero() -> Self { MpfrFloat(arena::push_zero()) }
    #[inline]
    fn is_zero(&self) -> bool { arena::with(self.0, |a| a.is_zero()) }
}

impl One for MpfrFloat {
    #[inline]
    fn one() -> Self { MpfrFloat(arena::push_one()) }
}

impl Default for MpfrFloat {
    #[inline]
    fn default() -> Self { Self::zero() }
}

#[derive(Debug, Clone)]
pub struct ParseMpfrError(String);

impl std::fmt::Display for ParseMpfrError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "MpfrFloat parse error: {}", self.0)
    }
}

impl std::error::Error for ParseMpfrError {}

impl Num for MpfrFloat {
    type FromStrRadixErr = ParseMpfrError;
    fn from_str_radix(s: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        RugFloat::parse_radix(s, radix as i32)
            .map(|incomplete| {
                MpfrFloat(arena::push(RugFloat::with_val(default_precision(), incomplete)))
            })
            .map_err(|e| ParseMpfrError(format!("{e}")))
    }
}

impl Signed for MpfrFloat {
    #[inline]
    fn abs(&self) -> Self {
        let val = arena::with(self.0, |a| a.clone().abs());
        MpfrFloat(arena::push(val))
    }
    #[inline]
    fn abs_sub(&self, other: &Self) -> Self {
        if self <= other { Self::zero() } else { *self - *other }
    }
    #[inline]
    fn signum(&self) -> Self {
        let (is_z, is_n) = arena::with(self.0, |a| (a.is_zero(), a.is_sign_negative()));
        if is_z { Self::zero() }
        else if is_n { -Self::one() }
        else { Self::one() }
    }
    #[inline]
    fn is_positive(&self) -> bool {
        arena::with(self.0, |a| !a.is_zero() && !a.is_sign_negative())
    }
    #[inline]
    fn is_negative(&self) -> bool {
        arena::with(self.0, |a| !a.is_zero() && a.is_sign_negative())
    }
}

impl FromPrimitive for MpfrFloat {
    fn from_i64(n: i64) -> Option<Self> { Some(MpfrFloat(arena::push(RugFloat::with_val(default_precision(), n)))) }
    fn from_u64(n: u64) -> Option<Self> { Some(MpfrFloat(arena::push(RugFloat::with_val(default_precision(), n)))) }
    fn from_isize(n: isize) -> Option<Self> { Self::from_i64(n as i64) }
    fn from_usize(n: usize) -> Option<Self> { Self::from_u64(n as u64) }
    fn from_i32(n: i32) -> Option<Self> { Self::from_i64(n as i64) }
    fn from_u32(n: u32) -> Option<Self> { Self::from_u64(n as u64) }
    fn from_f32(f: f32) -> Option<Self> { Some(MpfrFloat(arena::push(RugFloat::with_val(default_precision(), f)))) }
    fn from_f64(f: f64) -> Option<Self> { Some(MpfrFloat(arena::push(RugFloat::with_val(default_precision(), f)))) }
}

impl From<f64> for MpfrFloat { fn from(f: f64) -> Self { Self::from_f64(f).unwrap() } }
impl From<i64> for MpfrFloat { fn from(n: i64) -> Self { Self::from_i64(n).unwrap() } }
impl From<RugFloat> for MpfrFloat { fn from(f: RugFloat) -> Self { MpfrFloat(arena::push(f)) } }

impl std::fmt::Debug for MpfrFloat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        arena::with(self.0, |a| write!(f, "MpfrFloat({})", a))
    }
}

impl std::fmt::Display for MpfrFloat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        arena::with(self.0, |a| std::fmt::Display::fmt(a, f))
    }
}

impl std::fmt::LowerExp for MpfrFloat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        arena::with(self.0, |a| std::fmt::LowerExp::fmt(&a.to_f64(), f))
    }
}

impl crate::algebra::transcendental::BitWidthDiagnostic for MpfrFloat {
    #[inline]
    fn bit_width(&self) -> (u64, u64) {
        (arena::with(self.0, |a| a.prec() as u64), 0)
    }
}

impl num_traits::ToPrimitive for MpfrFloat {
    fn to_i64(&self) -> Option<i64> { Some(self.to_f64() as i64) }
    fn to_u64(&self) -> Option<u64> { Some(self.to_f64() as u64) }
    fn to_isize(&self) -> Option<isize> { Some(self.to_f64() as isize) }
    fn to_usize(&self) -> Option<usize> { Some(self.to_f64() as usize) }
    fn to_f64(&self) -> Option<f64> { Some(self.to_f64()) }
}

impl num_traits::NumCast for MpfrFloat {
    fn from<T: num_traits::ToPrimitive>(n: T) -> Option<Self> {
        n.to_f64().and_then(|v| Self::from_f64(v))
    }
}
