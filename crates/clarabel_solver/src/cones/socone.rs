use super::*;
use clarabel_algebra::*;

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
    pub u: Vec<T>,
    pub v: Vec<T>,
    //additional scalar terms for rank-2 rep
    d: T,
    pub η: T,
}

impl<T> SecondOrderCone<T>
where
    T: FloatT,
{
    pub fn new(dim: usize) -> Self {
        assert!(dim >= 2);
        Self {
            dim,
            w: vec![T::zero(); dim],
            λ: vec![T::zero(); dim],
            u: vec![T::zero(); dim],
            v: vec![T::zero(); dim],
            d: T::one(),
            η: T::zero(),
        }
    }
}

impl<T> Cone<T> for SecondOrderCone<T>
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
        δ.copy_from(e);
        δ.reciprocal();
        δ.scale(e.mean());

        false
    }

    fn WtW_is_diagonal(&self) -> bool {
        true
    }

    fn update_scaling(&mut self, s: &[T], z: &[T]) {

        let (z1, z2) = (z[0], &z[1..]);
        let (s1, s2) = (s[0], &s[1..]);

        let zscale = T::sqrt(z1 * z1 - z2.sumsq());
        let sscale = T::sqrt(s1 * s1 - s2.sumsq());

        let two = T::from(2.).unwrap();
        let half = T::recip(two);

        let gamma = T::sqrt((T::one() + s.dot(z) / (zscale * sscale)) * half);

        let w = &mut self.w;
        w.copy_from(s);
        w.scale(T::recip(two * sscale * gamma));
        w[0] += z[0] / (two * zscale * gamma);
        w[1..].axpby(-T::recip(two * zscale * gamma), &z[1..], T::one());

        //intermediate calcs for u,v,d,η
        let w0p1 = w[0] + T::one();
        let w1sq = w[1..].sumsq();
        let w0sq = w[0] * w[0];
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
        self.u[0] = u0;
        self.u[1..].axpby(u1, &self.w[1..], T::zero());
        self.v[0] = v0;
        self.v[1..].axpby(v1, &self.w[1..], T::zero());

        //λ = Wz.  Use inner function here because can't
        //borrow self and self.λ at the same time
        _soc_gemv_W_inner(&self.w, self.η, z, &mut self.λ, T::one(), T::zero());
    }

    fn set_identity_scaling(&mut self) {
        self.d = T::one();
        self.u.fill(T::zero());
        self.v.fill(T::zero());
        self.η = T::one();
        self.w.fill(T::zero());
    }

    fn λ_circ_λ(&self, x: &mut [T]) {
        self.circ_op(x, &self.λ, &self.λ);
    }

    fn circ_op(&self, x: &mut [T], y: &[T], z: &[T]) {
        x[0] = y.dot(z);
        let (y0, z0) = (y[0], z[0]);
        x[1..].waxpby(y0, &z[1..], z0, &y[1..]);
    }

    fn λ_inv_circ_op(&self, x: &mut [T], z: &[T]) {
        self.inv_circ_op(x, &self.λ, z);
    }

    fn inv_circ_op(&self, x: &mut [T], y: &[T], z: &[T]) {
        let p = y[0] * y[0] - y[1..].sumsq();
        let pinv = T::recip(p);
        let v = y[1..].dot(&z[1..]);

        x[0] = (y[0] * z[0] - v) * pinv;

        let c1 = pinv * (v / y[0] - z[0]);
        let c2 = T::recip(y[0]);
        x[1..].waxpby(c1, &y[1..], c2, &z[1..]);
    }

    fn shift_to_cone(&self, z: &mut [T]) {
        z[0] = T::max(z[0], T::zero());

        let α = z[0] * z[0] - z[1..].sumsq();

        if α < T::epsilon() {
            //done in two stages since otherwise (1.-α) = -α for
            //large α, which makes z exactly 0.0 (or worse, -0.0 )
            z[0] -= α;
            z[0] += T::one();
        }
    }

    fn get_WtW_block(&self, WtWblock: &mut [T]) {
        //NB: we are returning here the diagonal D block from the
        //sparse representation of W^TW, but not the
        //extra two entries at the bottom right of the block.
        WtWblock.fill(self.η * self.η);
        WtWblock[0] *= self.d;
    }

    fn gemv_W(&self, _is_transpose: MatrixShape, x: &[T], y: &mut [T], α: T, β: T) {

        // symmetric, so ignore transpose
        _soc_gemv_W_inner(&self.w, self.η, x, y, α, β);
    }

    fn gemv_Winv(&self, _is_transpose: MatrixShape, x: &[T], y: &mut [T], α: T, β: T) {

        // symmetric, so ignore transpose
        _soc_gemv_Winv_inner(&self.w, self.η, x, y, α, β);
    }

    fn add_scaled_e(&self, x: &mut [T], α: T) {
        //e is (1,0.0..0)
        x[0] += α;
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
    let four = T::from(4.).unwrap();

    let a = y[0] * y[0] - y[1..].sumsq();
    let b = two * (x[0] * y[0] - x[1..].dot(&y[1..]));
    let c = x[0] * x[0] - x[1..].sumsq(); //should be ≥0
    let d = b * b - four * a * c;

    if c < T::zero() {
        panic!("starting point of line search not in SOC");
    }

    let out;
    if (a > T::zero() && b > T::zero()) || (d < T::zero()) {
        // all negative roots / complex root pair
        // -> infinite step length
        out = T::max_value();
    } else {
        let sqrtd = T::sqrt(d);
        let mut r1 = (-b + sqrtd) / (two * a);
        let mut r2 = (-b - sqrtd) / (two * a);
        // return the minimum positive root
        if r1 < T::zero() {
            r1 = T::max_value();
        }
        if r2 < T::zero() {
            r2 = T::max_value();
        }
        out = T::min(r1, r2);
    }
    out
}

// We move the actual implementations of gemv_{W,Winv} outside
// here.  The operation λ = Wz produces a borrow conflict
// otherwise because λ is part of the cone's internal data
// and we can't borrow self and &mut λ at the same time.

#[allow(non_snake_case)]
fn _soc_gemv_W_inner<T>(w: &[T], η: T, x: &[T], y: &mut [T], α: T, β: T)
where
    T: FloatT,
{
    // use the fast product method from ECOS ECC paper
    let ζ = w[1..].dot(&x[1..]);
    let c = x[0] + ζ / (T::one() + w[0]);

    y[0] = (α * η) * (w[0] * x[0] + ζ) + β * y[0];

    y[1..].axpby(α * η * c, &w[1..], β);
    y[1..].axpby(α * η, &x[1..], T::one());
}

fn _soc_gemv_Winv_inner<T>(w: &[T], η: T, x: &[T], y: &mut [T], α: T, β: T)
where
    T: FloatT,
{
    // use the fast inverse product method from ECOS ECC paper
    let ζ = w[1..].dot(&x[1..]);
    let c = -x[0] + ζ / (T::one() + w[0]);

    y[0] = (α / η) * (w[0] * x[0] - ζ) + β * y[0];

    y[1..].axpby(α / η * c, &w[1..], β);
    y[1..].axpby(α / η, &x[1..], T::one());
}
