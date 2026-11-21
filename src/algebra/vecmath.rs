use super::{FloatT, ScalarMath, VectorMath};
use itertools::izip;
use std::borrow::Borrow;
use std::iter::zip;

impl<T: FloatT> VectorMath<T> for [T] {
    fn copy_from(&mut self, src: &[T]) -> &mut Self {
        for (d, s) in zip(&mut *self, src) {
            *d = s.clone();
        }
        self
    }

    fn select(&self, index: &[bool]) -> Vec<T> {
        assert_eq!(self.len(), index.len());
        zip(self, index)
            .filter(|(_x, &b)| b)
            .map(|(x, _b)| x.clone())
            .collect()
    }

    fn scalarop(&mut self, op: impl Fn(T) -> T) -> &mut Self {
        for x in &mut *self {
            *x = op(x.clone());
        }
        self
    }

    fn scalarop_from(&mut self, op: impl Fn(T) -> T, v: &[T]) -> &mut Self {
        for (x, v) in zip(&mut *self, v) {
            *x = op(v.clone());
        }
        self
    }

    fn translate(&mut self, c: T) -> &mut Self {
        //NB: translate is a scalar shift of all variables and is
        //used only in the NN cone to force vectors into R^n_+
        self.scalarop(|x| x + c.clone())
    }

    fn set(&mut self, c: T) -> &mut Self {
        for x in &mut *self {
            *x = c.clone();
        }
        self
    }

    fn scale(&mut self, c: T) -> &mut Self {
        self.scalarop(|x| x * c.clone())
    }

    fn recip(&mut self) -> &mut Self {
        self.scalarop(T::recip)
    }

    fn sqrt(&mut self) -> &mut Self {
        self.scalarop(T::sqrt)
    }

    fn rsqrt(&mut self) -> &mut Self {
        self.scalarop(|x| T::recip(T::sqrt(x)))
    }

    fn negate(&mut self) -> &mut Self {
        self.scalarop(|x| -x)
    }

    fn hadamard(&mut self, y: &[T]) -> &mut Self {
        zip(&mut *self, y).for_each(|(x, y)| *x *= y.clone());
        self
    }

    fn clip(&mut self, min_thresh: T, max_thresh: T) -> &mut Self {
        self.scalarop(|x| x.clip(min_thresh.clone(), max_thresh.clone()))
    }

    fn normalize(&mut self) -> T {
        let norm = self.norm();
        if norm.is_zero() {
            return T::zero();
        }
        self.scale(norm.clone().recip());
        norm
    }

    fn dot(&self, y: &[T]) -> T {
        zip(self, y).fold(T::zero(), |acc, (x, y)| acc + x.clone() * y.clone())
    }

    fn dot_shifted(z: &[T], s: &[T], dz: &[T], ds: &[T], α: T) -> T {
        assert_eq!(z.len(), s.len());
        assert_eq!(z.len(), dz.len());
        assert_eq!(s.len(), ds.len());

        let mut out = T::zero();
        for (s, ds, z, dz) in izip!(s, ds, z, dz) {
            let si = s.clone() + α.clone() * ds.clone();
            let zi = z.clone() + α.clone() * dz.clone();
            out += si * zi;
        }
        out
    }

    fn dist(&self, y: &Self) -> T {
        let dist2 = zip(self, y).fold(T::zero(), |acc, (x, y)| {
            acc + T::powi(x.clone() - y.clone(), 2)
        });
        T::sqrt(dist2)
    }

    fn sum(&self) -> T {
        self.iter().fold(T::zero(), |acc, x| acc + x.clone())
    }

    fn sumsq(&self) -> T {
        self.dot(self)
    }

    // 2-norm
    fn norm(&self) -> T {
        // T::sqrt(self.sumsq()) // not robust
        stable_norm(self.iter())
    }

    //2-norm of elementwise product self.*v
    fn norm_scaled(&self, v: &[T]) -> T {
        assert_eq!(self.len(), v.len());
        // `T: Borrow<T>`, so we can pass owned values (each from a
        // single .clone() pair) directly to stable_norm without a Vec
        // intermediate. Was per-call heap-alloc on f64 builds; matters
        // even more for RationalReal where the Vec<RationalReal>
        // intermediate would push N extra arena entries.
        stable_norm(izip!(self, v).map(|(yi, vi)| yi.clone() * vi.clone()))
    }

    // 2-norm of (self + α.dz)
    fn norm_shifted(&self, dz: &[T], α: T) -> T {
        // See `norm_scaled` — pass the mapped iterator directly.
        stable_norm(izip!(self, dz).map(move |(zi, dzi)| zi.clone() + α.clone() * dzi.clone()))
    }

    // Returns infinity norm
    fn norm_inf(&self) -> T {
        let mut out = T::zero();
        for v in self {
            if v.clone().is_nan() {
                return T::nan();
            }
            out = T::max(out, v.clone().abs());
        }
        out
    }

    // Returns one norm
    fn norm_one(&self) -> T {
        self.iter().fold(T::zero(), |acc, v| acc + v.clone().abs())
    }

    //inf-norm of elementwise product self.*v
    fn norm_inf_scaled(&self, v: &Self) -> T {
        assert_eq!(self.len(), v.len());
        zip(self, v).fold(T::zero(), |acc, (x, y)| {
            T::max(acc, (x.clone() * y.clone()).abs())
        })
    }

    //
    fn norm_one_scaled(&self, v: &Self) -> T {
        zip(self, v).fold(T::zero(), |acc, (x, y)| {
            acc + (x.clone() * y.clone()).abs()
        })
    }

    // max absolute difference (used for unit testing)
    fn norm_inf_diff(&self, b: &[T]) -> T {
        zip(self, b).fold(T::zero(), |acc, (x, y)| {
            T::max(acc, (x.clone() - y.clone()).abs())
        })
    }

    fn minimum(&self) -> T {
        self.iter()
            .fold(T::infinity(), |r, s| T::min(r, s.clone()))
    }

    fn maximum(&self) -> T {
        self.iter()
            .fold(-T::infinity(), |r, s| T::max(r, s.clone()))
    }

    fn mean(&self) -> T {
        let mean = if self.is_empty() {
            T::zero()
        } else {
            let num = self.iter().fold(T::zero(), |r, s| r + s.clone());
            let den = T::from_usize(self.len()).unwrap();
            num / den
        };
        mean
    }

    fn is_finite(&self) -> bool {
        self.iter().all(|x| T::is_finite(x.clone()))
    }

    fn axpby(&mut self, a: T, x: &[T], b: T) -> &mut Self {
        assert_eq!(self.len(), x.len());

        zip(&mut *self, x).for_each(|(y, x)| {
            *y = a.clone() * x.clone() + b.clone() * y.clone();
        });
        self
    }

    fn waxpby(&mut self, a: T, x: &[T], b: T, y: &[T]) -> &mut Self {
        assert_eq!(self.len(), x.len());
        assert_eq!(self.len(), y.len());

        for (w, (x, y)) in zip(&mut *self, zip(x, y)) {
            *w = a.clone() * x.clone() + b.clone() * y.clone();
        }
        self
    }
}

// numerically more stable 2-norm that avoids overflow/underflow
fn stable_norm<T, I, B>(x: I) -> T
where
    T: FloatT,
    I: Iterator<Item = B>,
    B: Borrow<T>,
{
    let (scale, sumsq) =
        x.filter(|b| !b.borrow().is_zero())
            .fold((T::zero(), T::one()), |(scale, sumsq), b| {
                let xi = b.borrow().clone();
                let absxi = xi.abs();
                if scale < absxi {
                    let r = scale / absxi.clone();
                    (absxi, T::one() + sumsq * r.clone() * r)
                } else {
                    let r = absxi / scale.clone();
                    (scale, sumsq + r.clone() * r)
                }
            });
    scale * sumsq.sqrt()
}
