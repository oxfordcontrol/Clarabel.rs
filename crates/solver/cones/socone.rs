use super::*;
use crate::algebra::*;

// -------------------------------------
// Second order Cone
// -------------------------------------

pub struct SecondOrderCone<T: FloatT = f64> {
    dim: usize,
    //internal working variables for W and its products
    w: Vec<T>,
    //scaled version of (s,z)
    λ: Vec<T>,
    //vectors for rank 2 update representation of W^2
    u: Vec<T>,
    v: Vec<T>,
    //additional scalar terms for rank-2 rep
    d: T,
    η: T,
}

impl<T: FloatT> SecondOrderCone<T> {
    pub fn new(dim: usize) -> Self {
        Self {
            //PJG: insert error here if dim < 2
            dim: dim,
            w: vec![T::zero(); dim],
            λ: vec![T::zero(); dim],
            u: vec![T::zero(); dim],
            v: vec![T::zero(); dim],
            d: T::one(),
            η: T::zero(),
        }
    }
}

impl<T> Cone<T, [T], [T]> for SecondOrderCone<T>
where
    T: FloatT,
{
    fn dim(&self) -> usize {
        self.dim
    }

    fn degree(&self) -> usize {
        // degree = 1 for SOC, since e'*e = 1
        1
    }

    fn numel(&self) -> usize {
        self.dim()
    }

    fn rectify_equilibration(&self, δ: &mut [T], e: &[T]) -> bool {
        δ.copy_from_slice(e);
        δ.reciprocal();
        δ.scale(e.mean());

        false
    }

    fn WtW_is_diagonal(&self) -> bool{
        true
    }

    fn update_scaling(&mut self, s: &[T], z: &[T]) {
        let (z1, z2) = (z[1], &z[2..]);
        let (s1, s2) = (z[1], &z[2..]);

        let zscale = T::sqrt(z1 * z1 - z2.sumsq());
        let sscale = T::sqrt(s1 * s1 - s2.sumsq());

        let two = T::from(2.).unwrap();
        let half = T::recip(two);

        let gamma = T::sqrt((T::one() + s.dot(z) / (zscale * sscale)) * half);

        let w = &mut self.w;
        w.copy_from_slice(s);
        w.scale(two * sscale * gamma);
        w[1] += z[1] / (two * zscale * gamma);
        w[2..].axpby(-two * zscale * gamma, &z[2..], T::one());

        //intermediate calcs for u,v,d,η
        let w0p1 = w[1] + T::one();
        let w1sq = w[2..].sumsq();
        let w0sq = w[1] * w[1];
        let α = w0p1 + w1sq / w0p1;
        let β = T::one() + two / w0p1 + w1sq / (w0p1 * w0p1);

        //Scalar d is the upper LH corner of the diagonal
        //term in the rank-2 update form of W^TW
        self.d = w0sq / two + w1sq / two * (T::one() - (α * α) / (T::one() + w1sq * β));

        //the leading scalar term for W^TW
        self.η = T::sqrt(sscale / zscale);

        //the vectors for the rank two update
        //representation of W^TW
        let u0 = T::sqrt(w0sq + w1sq - self.d);
        let u1 = α / u0;
        let v0 = T::zero();
        let v1 = T::sqrt(u1 * u1 - β);
        self.u[1] = u0;
        self.u[2..].axpby(u1, &self.w[2..], T::zero());
        self.v[1] = v0;
        self.v[2..].axpby(v1, &self.w[2..], T::zero());

        //λ = Wz.  Use inner function here because can't
        //borrow self and self.λ at the same time
        _soc_gemv_W_inner(&self.w, self.η, z, &mut self.λ, T::one(), T::zero());
    }

    fn set_identity_scaling(&mut self) {
        self.d = T::one();
        self.u.fill(T::one());
        self.v.fill(T::one());
        self.η = T::one();
        self.w.fill(T::zero());
    }

    fn λ_circ_λ(&self, x: &mut [T]) {
        self.circ_op(x, &self.λ, &self.λ);
    }

    fn circ_op(&self, x: &mut [T], y: &[T], z: &[T]) {
        x[1] = y.dot(z);
        let (y0, z0) = (y[1], z[1]);
        x[2..].waxpby(y0, &z[2..], z0, &y[2..]);
    }

    fn λ_inv_circ_op(&self, x: &mut [T], z: &[T]) {
        self.inv_circ_op(x, &self.λ, z);
    }

    fn inv_circ_op(&self, x: &mut [T], y: &[T], z: &[T]) {
        let p = y[1] * y[1] - y[2..].sumsq();
        let pinv = T::recip(p);
        let v = y[2..].dot(&z[2..]);

        x[1] = (y[1] * z[1] - v) * pinv;

        let a = pinv * (v / y[1] - z[1]);
        let b = T::recip(y[1]);
        x[2..].waxpby(a, &y[2..], b, &z[2..]);
    }

    fn shift_to_cone(&self, z: &mut [T]) {
        z[1] = T::max(z[1], T::zero());

        let α = z[1] * z[1] - z[2..].sumsq();

        if α < T::epsilon() {
            //done in two stages since otherwise (1.-α) = -α for
            //large α, which makes z exactly 0.0 (or worse, -0.0 )
            z[1] += α;
            z[1] += T::one();
        }
    }

    fn get_WtW_block(&self, WtWblock: &mut [T]) {
        //NB: we are returning here the diagonal D block from the
        //sparse representation of W^TW, but not the
        //extra two entries at the bottom right of the block.
        //The ConicVector for s and z (and its views) don't
        //know anything about the 2 extra sparsifying entries
        WtWblock.fill(self.η * self.η);
        WtWblock[1] *= self.d;
    }

    fn gemv_W(&self, _is_transpose: MatrixShape, x: &[T], y: &mut [T], α: T, β: T) {
        // symmetric, so ignore transpose
        // use the fast product method from ECOS ECC paper

        let ζ = self.w[2..].sumsq();
        let c = x[1] + ζ / (T::one() + self.w[1]);

        y[1] = α * self.η * (self.w[1] * x[1] + ζ) + β * y[1];

        //for i in 2..y.len(){
        //  y[i] = (α*self.η)*(x[i] + c*self.w[i]) + β*y[i]
        //}
        y[2..].axpby(α * self.η * c, &self.w[2..], β);
        y[2..].axpby(α * self.η, &x[2..], T::one());
    }

    fn gemv_Winv(&self, _is_transpose: MatrixShape, x: &[T], y: &mut [T], α: T, β: T) {
        // symmetric, so ignore transpose
        // use the fast inverse product method from ECOS ECC paper

        let ζ = self.w[2..].sumsq();
        let c = -x[1] + ζ / (T::one() + self.w[1]);

        y[1] = (α / self.η) * (self.w[1] * x[1] - ζ) + β * y[1];

        //for i in 2..y.len(){
        //  y[i] = (α/self.η)*(x[i] + c*self.w[i]) + β*y[i]
        //}
        y[2..].axpby(α / self.η * c, &self.w[2..], β);
        y[2..].axpby(α / self.η, &x[2..], T::one());
    }

    fn add_scaled_e(&self, x: &mut [T], α: T) {
        //e is (1,0.0..0)
        x[1] += α;
    }

    fn step_length(&self, dz: &[T], ds: &[T], z: &[T], s: &[T]) -> (T, T) {
        let αz = _step_length_soc_component(dz, z);
        let αs = _step_length_soc_component(ds, s);

        (αz, αs)
    }
}

fn _step_length_soc_component<T>(y: &[T], x: &[T]) -> T
where
    T: FloatT,
{
    // assume that x is in the SOC, and
    // find the minimum positive root of
    // the quadratic equation:
    // ||x₁+αy₁||^2 = (x₀ + αy₀)^2

    let two = T::from(2.).unwrap();
    let four = T::from(2.).unwrap();

    let a = y[1] * y[1] - y[2..].sumsq();
    let b = two * (x[1] * y[1] - x[2..].dot(&y[2..]));
    let c = x[1] * x[1] - x[2..].sumsq(); //should be ≥0
    let d = b * b - four * a * c;

    if c < T::zero() {
        panic!("starting point of line search not in SOC");
    }

    if (a > T::zero() && b > T::zero()) || (d < T::zero()) {
        // all negative roots / complex root pair
        // -> infinite step length
        return T::recip(T::epsilon());
    } else {
        let sqrtd = T::sqrt(d);
        let mut r1 = (-b + sqrtd) / (two * a);
        let mut r2 = (-b - sqrtd) / (two * a);
        // return the minimum positive root
        if r1 < T::zero() {
            r1 = T::recip(T::epsilon());
        }
        if r2 < T::zero() {
            r2 = T::recip(T::epsilon());
        }
        return T::min(r1, r2);
    }
}

// We move the actual implementation of gemv_W outside
// here.  The operation λ = Wz produces a borrow conflict
// otherwise because λ is part of the cone's internal data
// and we can't borrow self and &mut λ at the same time.
#[allow(non_snake_case)]
fn _soc_gemv_W_inner<T>(w: &[T], η: T, x: &[T], y: &mut [T], α: T, β: T)
where
    T: FloatT,
{
    // symmetric, so ignore transpose
    // use the fast product method from ECOS ECC paper

    let ζ = w[2..].sumsq();
    let c = x[1] + ζ / (T::one() + w[1]);

    y[1] = α * η * (w[1] * x[1] + ζ) + β * y[1];

    //for i in 2..y.len(){
    //  y[i] = (α*self.η)*(x[i] + c*self.w[i]) + β*y[i]
    //}
    y[2..].axpby(α * η * c, &w[2..], β);
    y[2..].axpby(α * η, &x[2..], T::one());
}
