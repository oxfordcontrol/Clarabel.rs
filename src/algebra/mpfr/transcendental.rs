use super::arena;
use super::real::MpfrFloat;
use crate::algebra::transcendental::Transcendental;
use rug::ops::Pow;
use rug::Float as RugFloat;

impl Transcendental for MpfrFloat {
    fn sqrt(self) -> Self {
        let val = arena::with(self.0, |a| a.clone().sqrt());
        MpfrFloat(arena::push(val))
    }
    fn ln(self) -> Self {
        let val = arena::with(self.0, |a| a.clone().ln());
        MpfrFloat(arena::push(val))
    }
    fn exp(self) -> Self {
        let val = arena::with(self.0, |a| a.clone().exp());
        MpfrFloat(arena::push(val))
    }
    fn powf(self, n: Self) -> Self {
        let val = arena::with2(self.0, n.0, |a, b| a.clone().pow(b));
        MpfrFloat(arena::push(val))
    }
    fn powi(self, n: i32) -> Self {
        let val = arena::with(self.0, |a| a.clone().pow(n));
        MpfrFloat(arena::push(val))
    }
    fn recip(self) -> Self {
        let val = arena::with(self.0, |a| {
            let p = a.prec();
            let one = RugFloat::with_val(p, 1);
            one / a
        });
        MpfrFloat(arena::push(val))
    }
    fn sin(self) -> Self {
        let val = arena::with(self.0, |a| a.clone().sin());
        MpfrFloat(arena::push(val))
    }
    fn cos(self) -> Self {
        let val = arena::with(self.0, |a| a.clone().cos());
        MpfrFloat(arena::push(val))
    }
    fn atan2(self, x: Self) -> Self {
        let val = arena::with2(self.0, x.0, |a, b| a.clone().atan2(b));
        MpfrFloat(arena::push(val))
    }
}
