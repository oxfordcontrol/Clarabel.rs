use super::{FloatT, ScalarMath, VectorMath};
use itertools::izip;
use std::iter::zip;

impl<T: FloatT> VectorMath for [T] {
    type T = T;
    fn copy_from(&mut self, src: &[T]) -> &mut Self {
        self.copy_from_slice(src);
        self
    }

    fn select(&self, index: &[bool]) -> Vec<T> {
        assert_eq!(self.len(), index.len());
        zip(self, index)
            .filter(|(_x, &b)| b)
            .map(|(&x, _b)| x)
            .collect()
    }

    fn scalarop(&mut self, op: impl Fn(T) -> T) -> &mut Self {
        for x in &mut *self {
            *x = op(*x);
        }
        self
    }

    fn scalarop_from(&mut self, op: impl Fn(T) -> T, v: &[T]) -> &mut Self {
        for (x, v) in zip(&mut *self, v) {
            *x = op(*v);
        }
        self
    }

    fn translate(&mut self, c: T) -> &mut Self {
        //NB: translate is a scalar shift of all variables and is
        //used only in the NN cone to force vectors into R^n_+
        self.scalarop(|x| x + c)
    }

    fn set(&mut self, c: T) -> &mut Self {
        self.scalarop(|_x| c)
    }

    fn scale(&mut self, c: T) -> &mut Self {
        self.scalarop(|x| x * c)
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
        zip(&mut *self, y).for_each(|(x, y)| *x *= *y);
        self
    }

    fn clip(&mut self, min_thresh: T, max_thresh: T) -> &mut Self {
        self.scalarop(|x| x.clip(min_thresh, max_thresh))
    }

    fn normalize(&mut self) -> T {
        let norm = self.norm();
        if norm == T::zero() {
            return T::zero();
        }
        self.scale(norm.recip());
        norm
    }

    //PJG: dot, dot_shifted and sum could be rewritten with a common accumulator function
    //and a common base base parameter
    fn dot(&self, y: &[T]) -> T {
        let iter = zip(self, y);
        let op = |(&x, &y)| x * y;
        accumulate_pairwise(iter, op)
    }

    fn dot_shifted(z: &[T], s: &[T], dz: &[T], ds: &[T], α: T) -> T {
        assert_eq!(z.len(), s.len());
        assert_eq!(z.len(), dz.len());
        assert_eq!(s.len(), ds.len());

        let iter = izip!(s, ds, z, dz);
        let op = |(&s, &ds, &z, &dz)| {
            let si = s + α * ds;
            let zi = z + α * dz;
            si * zi
        };
        accumulate_pairwise(iter, op)
    }

    fn dist(&self, y: &Self) -> T {
        let iter = zip(self, y);
        let op = |(&x, &y)| T::powi(x - y, 2);
        let dist2 = accumulate_pairwise(iter, op);
        T::sqrt(dist2)
    }

    fn sum(&self) -> T {
        accumulate_pairwise(self.iter(), |&x| x)
    }

    fn sumsq(&self) -> T {
        self.dot(self)
    }

    // 2-norm
    fn norm(&self) -> T {
        T::sqrt(self.sumsq())
    }

    // Returns infinity norm
    fn norm_inf(&self) -> T {
        let mut out = T::zero();
        for v in self.iter().map(|v| v.abs()) {
            if v.is_nan() {
                return T::nan();
            }
            out = if v > out { v } else { out };
        }
        out
    }

    // Returns one norm
    fn norm_one(&self) -> T {
        accumulate_pairwise(self.iter(), |&x| x.abs())
    }

    //PJG: use generic accumulator here
    //two-norm of elementwise product self.*v
    fn norm_scaled(&self, v: &[T]) -> T {
        assert_eq!(self.len(), v.len());

        let iter = zip(self, v);
        let op = |(&x, &y)| {
            let prod = x * y;
            prod * prod
        };

        let total = accumulate_pairwise(iter, op);
        T::sqrt(total)
    }

    //inf-norm of elementwise product self.*v
    fn norm_inf_scaled(&self, v: &Self) -> Self::T {
        assert_eq!(self.len(), v.len());
        zip(self, v).fold(T::zero(), |acc, (&x, &y)| T::max(acc, T::abs(x * y)))
    }

    //
    fn norm_one_scaled(&self, v: &Self) -> Self::T {
        let iter = zip(self, v);
        let op = |(&x, &y)| T::abs(x * y);
        accumulate_pairwise(iter, op)
    }

    // max absolute difference (used for unit testing)
    fn norm_inf_diff(&self, b: &[T]) -> T {
        zip(self, b).fold(T::zero(), |acc, (x, y)| T::max(acc, T::abs(*x - *y)))
    }

    fn minimum(&self) -> T {
        self.iter().fold(T::infinity(), |r, &s| T::min(r, s))
    }

    fn maximum(&self) -> T {
        self.iter().fold(-T::infinity(), |r, &s| T::max(r, s))
    }

    fn mean(&self) -> T {
        let mean = if self.is_empty() {
            T::zero()
        } else {
            let num = self.sum();
            let den = T::from_usize(self.len()).unwrap();
            num / den
        };
        mean
    }

    fn is_finite(&self) -> bool {
        self.iter().all(|&x| T::is_finite(x))
    }

    fn axpby(&mut self, a: T, x: &[T], b: T) -> &mut Self {
        assert_eq!(self.len(), x.len());

        zip(&mut *self, x).for_each(|(y, x)| *y = a * (*x) + b * (*y));
        self
    }

    fn waxpby(&mut self, a: T, x: &[T], b: T, y: &[T]) -> &mut Self {
        assert_eq!(self.len(), x.len());
        assert_eq!(self.len(), y.len());

        for (w, (x, y)) in zip(&mut *self, zip(x, y)) {
            *w = a * (*x) + b * (*y);
        }
        self
    }
}

// ---------------------------------------------------------------------
// generic pairwise accumulator utility for sums, dot products etc

fn accumulate_pairwise<T, I, A, F>(x: I, op: F) -> T
where
    T: FloatT,
    I: IntoIterator<Item = A> + Clone,
    I::IntoIter: ExactSizeIterator,
    F: Fn(A) -> T,
{
    const BASE_CASE_DIM: usize = 16;

    let n = x.clone().into_iter().len();
    return if n == 0 {
        T::zero()
    } else {
        accumulate_pairwise_inner(x, &op, 0, n)
    };

    fn accumulate_pairwise_inner<T, I, A, F>(x: I, op: &F, i1: usize, n: usize) -> T
    where
        T: FloatT,
        I: IntoIterator<Item = A> + Clone,
        I::IntoIter: ExactSizeIterator,
        F: Fn(A) -> T,
    {
        if n < BASE_CASE_DIM {
            return x
                .into_iter()
                .skip(i1)
                .take(n)
                .fold(T::zero(), |acc, x| acc + op(x));
        } else {
            let n2 = n / 2;
            println!("n2 = {}", n2);
            return accumulate_pairwise_inner(x.clone(), op, i1, n2)
                + accumulate_pairwise_inner(x, op, i1 + n2, n - n2);
        }
    }
}

#[test]
fn test_dot_product() {
    let x = vec![1., 2., 3., 4.];
    let y = vec![4., 5., 6., 7.];
    assert_eq!(x.dot(&y), 60.);
}

#[test]
fn test_mean() {
    let x = vec![1., 2., 3., 4., 5.];
    assert_eq!(x.mean(), 3.);
    assert_eq!(x[0..1].mean(), 1.);
    assert_eq!(x[0..0].mean(), 0.);

    //taking the mean of a huge number of f32s is inaccurate for
    //naive summation, but the pairwise method should still work
    let n = 10000000usize;
    let x = vec![1.5f32; n];
    let mean = x.mean();
    assert_eq!(mean, 1.5f32);

    //example should be such that naive summation would
    //have been inaccurate.  'mean' this way is ~1.72
    let mean = x.iter().fold(0.0, |acc, &z| acc + z) / (n as f32);
    assert!((mean - 1.5f32).abs() > 0.2f32);
}

#[test]
fn test_sum() {
    let maxlen = 128 * 7 + 1; //awkward length to test base case
    let x: Vec<f64> = (1..=maxlen).map(|x| x as f64).collect();

    for i in 0..=x.len() {
        let z = &x[0..i];
        let sum1 = z.iter().fold(0.0, |acc, &z| acc + z);
        let sum2 = z.sum();
        assert_eq!(sum1, sum2);
    }
}

#[test]
fn test_dot() {
    let maxlen = 128 * 7 + 1; //awkward length to test base case
    let x: Vec<f64> = (1..=maxlen).map(|x| x as f64).collect();
    let y: Vec<f64> = (1..=maxlen)
        .map(|y| (y as f64 - 3.0) / 2.0 as f64)
        .collect();

    for i in 0..=x.len() {
        let xt = &x[0..i];
        let yt = &y[0..i];
        let dot1 = zip(xt, yt).fold(0.0, |acc, (&x, &y)| acc + x * y);
        let dot2 = xt.dot(yt);
        assert_eq!(dot1, dot2);
    }
}

#[test]
fn test_dot_shifted() {
    let maxlen = 128 * 7 + 1; //awkward length to test base case
    let z: Vec<f64> = (1..=maxlen).map(|z| z as f64).collect();
    let s: Vec<f64> = (1..=maxlen)
        .map(|s| (s as f64 - 3.0) / 2.0 as f64)
        .collect();

    let dz = vec![1.0; z.len()];
    let ds = vec![0.5; s.len()];
    let α = 0.5;

    for i in 0..=z.len() {
        let zt = &z[0..i];
        let st = &s[0..i];
        let dzt = &dz[0..i];
        let dst = &ds[0..i];
        let dot1 = <[f64] as VectorMath>::dot_shifted(zt, st, dzt, dst, α);
        let dot2 = zt.dot(st) + α * zt.dot(dst) + α * st.dot(dzt) + α * α * dzt.dot(dst);
        assert_eq!(dot1, dot2);
    }
}
