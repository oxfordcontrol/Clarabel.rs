use super::*;
use crate::{
    algebra::*,
    solver::{core::ScalingStrategy, CoreSettings},
};
use itertools::izip;

// -------------------------------------
// Second order Cone
// -------------------------------------

pub struct SecondOrderConeSparseData<T> {
    //vectors for rank 2 update representation of W^2
    pub u: Vec<T>,
    pub v: Vec<T>,

    //additional scalar terms for rank-2 rep
    pub d: T,
}

impl<T> SecondOrderConeSparseData<T>
where
    T: FloatT,
{
    pub fn new(dim: usize) -> Self {
        Self {
            u: vec![T::zero(); dim],
            v: vec![T::zero(); dim],
            d: T::one(),
        }
    }
}

pub struct SecondOrderCone<T> {
    pub dim: usize,
    //internal working variables for W and its products
    pub w: Vec<T>,
    //scaled version of (s,z)
    pub λ: Vec<T>,
    pub η: T,
    pub sparse_data: Option<SecondOrderConeSparseData<T>>,
}

impl<T> SecondOrderCone<T>
where
    T: FloatT,
{
    pub fn new(dim: usize) -> Self {
        const SOC_NO_EXPANSION_MAX_SIZE: usize = 4;

        assert!(dim >= 2);

        let w = vec![T::zero(); dim];
        let λ = vec![T::zero(); dim];
        let η = T::zero();

        let sparse_data = {
            if dim > SOC_NO_EXPANSION_MAX_SIZE {
                Some(SecondOrderConeSparseData::new(dim))
            } else {
                None
            }
        };

        Self {
            dim,
            w,
            λ,
            η,
            sparse_data,
        }
    }
}

impl<T> Cone<T> for SecondOrderCone<T>
where
    T: FloatT,
{
    fn degree(&self) -> usize {
        // degree = 1 for SOC, since e'*e = 1
        1
    }

    fn numel(&self) -> usize {
        self.dim
    }

    fn is_symmetric(&self) -> bool {
        true
    }

    fn is_sparse_expandable(&self) -> bool {
        self.sparse_data.is_some()
    }

    fn allows_primal_dual_scaling(&self) -> bool {
        true
    }

    fn rectify_equilibration(&self, δ: &mut [T], e: &[T]) -> bool {
        δ.copy_from(e).recip().scale(e.mean());

        true // scalar equilibration
    }

    // functions relating to unit vectors and cone initialization
    fn margins(&mut self, z: &mut [T], _pd: PrimalOrDualCone) -> (T, T) {
        let α = z[0] - z[1..].norm();
        let β = T::max(T::zero(), α);
        (α, β)
    }

    fn scaled_unit_shift(&self, z: &mut [T], α: T, _pd: PrimalOrDualCone) {
        z[0] += α;
    }

    fn unit_initialization(&self, z: &mut [T], s: &mut [T]) {
        s.fill(T::zero());
        z.fill(T::zero());
        self.scaled_unit_shift(s, T::one(), PrimalOrDualCone::PrimalCone);
        self.scaled_unit_shift(z, T::one(), PrimalOrDualCone::DualCone);
    }

    fn set_identity_scaling(&mut self) {
        self.w.fill(T::zero());
        self.w[0] = T::one();
        self.η = T::one();

        if let Some(sparse_data) = &mut self.sparse_data {
            sparse_data.d = (0.5).as_T();
            sparse_data.u.fill(T::zero());
            sparse_data.u[0] = T::FRAC_1_SQRT_2();
            sparse_data.v.fill(T::zero());
        }
    }

    fn update_scaling(
        &mut self,
        s: &[T],
        z: &[T],
        _μ: T,
        _scaling_strategy: ScalingStrategy,
    ) -> bool {
        let two: T = (2.0).as_T();
        let half: T = (0.5).as_T();

        //first calculate the scaled vector w
        let zscale = _sqrt_soc_residual(z);
        let sscale = _sqrt_soc_residual(s);

        //Fail if either s or z is not an interior point
        if zscale.is_zero() || sscale.is_zero() {
            return false;
        }

        //the leading scalar term for W^TW
        self.η = T::sqrt(sscale / zscale);

        // construct w and normalize
        let w = &mut self.w;
        w.copy_from(s);
        w.scale(sscale.recip());
        w[0] += z[0] / zscale;
        w[1..].axpby(-zscale.recip(), &z[1..], T::one());

        let wscale = _sqrt_soc_residual(w);
        // Fail if w is not an interior point
        if wscale.is_zero() {
            return false;
        }
        w.scale(wscale.recip());

        // try to force badly scaled w to come out normalized
        let w1sq = w[1..].sumsq();
        w[0] = T::sqrt(T::one() + w1sq);

        //---------------------
        //DEBUG ALTERNATIVE λ

        //Compute the scaling point λ.   Should satisfy λ = Wz = W^{-T}s
        let γ = half * wscale;
        self.λ[0] = γ;
        self.λ[1..].waxpby(
            (γ + z[0] / zscale) / sscale,
            &s[1..],
            (γ + s[0] / sscale) / zscale,
            &z[1..],
        );
        self.λ[1..].scale(T::recip(s[0] / sscale + z[0] / zscale + two * γ));
        self.λ.scale(T::sqrt(sscale * zscale));

        if let Some(sparse_data) = &mut self.sparse_data {
            //various intermediate calcs for u,v,d,η
            let α = two * w[0];

            //Scalar d is the upper LH corner of the diagonal
            //term in the rank-2 update form of W^TW
            let wsq = w[0] * w[0] + w1sq;
            let wsqinv = wsq.recip();
            sparse_data.d = half * wsqinv;

            //the vectors for the rank two update
            //representation of W^TW
            let u0 = T::sqrt(wsq - sparse_data.d);
            let u1 = α / u0;
            let v0 = T::zero();
            let v1 = T::sqrt(two * (two + wsqinv) / (two * wsq - wsqinv));

            sparse_data.u[0] = u0;
            sparse_data.u[1..].axpby(u1, &self.w[1..], T::zero());
            sparse_data.v[0] = v0;
            sparse_data.v[1..].axpby(v1, &self.w[1..], T::zero());
        }

        true
    }

    fn Hs_is_diagonal(&self) -> bool {
        self.is_sparse_expandable()
    }

    fn get_Hs(&self, Hsblock: &mut [T]) {
        if let Some(sparse_data) = &self.sparse_data {
            // For sparse form, we are returning here the diagonal D block
            // from the sparse representation of W^TW, but not the
            // extra two entries at the bottom right of the block.
            // The ConicVector for s and z (and its views) don't
            // know anything about the 2 extra sparsifying entries
            Hsblock.fill(self.η * self.η);
            Hsblock[0] *= sparse_data.d;
        } else {
            let two: T = (2.).as_T();
            // for dense form, we return H = \eta^2 (2*ww^T - J), where
            // J = diag(1,-I).  We are packing into dense triu form
            Hsblock[0] = two * self.w[0] * self.w[0] - T::one();
            let mut hidx = 1;

            for col in 1..self.dim {
                let wcol = self.w[col];
                for row in 0..=col {
                    Hsblock[hidx] = two * self.w[row] * wcol;
                    hidx += 1
                }
                //go back to add the offset term from J
                Hsblock[hidx - 1] += T::one()
            }
            Hsblock.scale(self.η * self.η);
        }
    }

    fn mul_Hs(&mut self, y: &mut [T], x: &[T], _work: &mut [T]) {
        //self.mul_W(MatrixShape::N, work, x, T::one(), T::zero()); // work = Wx
        //self.mul_W(MatrixShape::T, y, work, T::one(), T::zero()); // y = c Wᵀwork = W^TWx
        let c = self.w.dot(x) * (2.).as_T();
        y.copy_from(x);
        y[0] = -x[0];
        y.axpby(c, &self.w, T::one());
        y.scale(self.η * self.η);
    }

    fn affine_ds(&self, ds: &mut [T], _s: &[T]) {
        _circ_op(ds, &self.λ, &self.λ);
    }

    fn combined_ds_shift(&mut self, shift: &mut [T], step_z: &mut [T], step_s: &mut [T], σμ: T) {
        self._combined_ds_shift_symmetric(shift, step_z, step_s, σμ);
    }

    fn Δs_from_Δz_offset(&mut self, out: &mut [T], ds: &[T], _work: &mut [T], z: &[T]) {
        // out = Wᵀ(λ \ ds).  Below is equivalent,
        // but appears to be a little more stable

        let resz = _soc_residual(z);

        let λ1ds1 = self.λ[1..].dot(&ds[1..]);
        let w1ds1 = self.w[1..].dot(&ds[1..]);

        out.scalarop_from(|zi| -zi, z);
        out[0] = z[0];

        let c = self.λ[0] * ds[0] - λ1ds1;
        out.scale(c / resz);

        out[0] += self.η * w1ds1;
        for (outi, &dsi, &wi) in izip!(out[1..].iter_mut(), &ds[1..], &self.w[1..]) {
            *outi += self.η * (dsi + w1ds1 / (T::one() + self.w[0]) * wi);
        }

        out.scale(self.λ[0].recip());
    }

    fn step_length(
        &mut self,
        dz: &[T],
        ds: &[T],
        z: &[T],
        s: &[T],
        _settings: &CoreSettings<T>,
        αmax: T,
    ) -> (T, T) {
        let αz = _step_length_soc_component(z, dz, αmax);
        let αs = _step_length_soc_component(s, ds, αmax);

        (αz, αs)
    }

    fn compute_barrier(&mut self, z: &[T], s: &[T], dz: &[T], ds: &[T], α: T) -> T {
        let res_s = _soc_residual_shifted(s, ds, α);
        let res_z = _soc_residual_shifted(z, dz, α);

        // avoid numerical issue if res_s <= 0 or res_z <= 0
        if res_s > T::zero() && res_z > T::zero() {
            -(res_s * res_z).logsafe() * (0.5).as_T()
        } else {
            T::infinity()
        }
    }
}

// ---------------------------------------------
// operations supported by symmetric cones only
// ---------------------------------------------

impl<T> SymmetricCone<T> for SecondOrderCone<T>
where
    T: FloatT,
{
    fn λ_inv_circ_op(&mut self, x: &mut [T], z: &[T]) {
        _inv_circ_op(x, &self.λ, z);
    }

    fn mul_W(&mut self, _is_transpose: MatrixShape, y: &mut [T], x: &[T], α: T, β: T) {
        // symmetric, so ignore transpose
        _soc_mul_W_inner(y, x, α, β, &self.w, self.η);
    }

    fn mul_Winv(&mut self, _is_transpose: MatrixShape, y: &mut [T], x: &[T], α: T, β: T) {
        _soc_mul_Winv_inner(y, x, α, β, &self.w, self.η);
    }
}

// ---------------------------------------------
// Jordan algebra operations for symmetric cones
// ---------------------------------------------

impl<T> JordanAlgebra<T> for SecondOrderCone<T>
where
    T: FloatT,
{
    fn circ_op(&mut self, x: &mut [T], y: &[T], z: &[T]) {
        _circ_op(x, y, z);
    }

    fn inv_circ_op(&mut self, x: &mut [T], y: &[T], z: &[T]) {
        _inv_circ_op(x, y, z);
    }
}

// circ ops don't use self for this cone, so put the actual
// implementations outside so that they can be called by
// other functions with entering borrow check hell

fn _circ_op<T>(x: &mut [T], y: &[T], z: &[T])
where
    T: FloatT,
{
    x[0] = y.dot(z);
    let (y0, z0) = (y[0], z[0]);
    x[1..].waxpby(y0, &z[1..], z0, &y[1..]);
}

fn _inv_circ_op<T>(x: &mut [T], y: &[T], z: &[T])
where
    T: FloatT,
{
    let p = _soc_residual(y);
    let pinv = T::recip(p);
    let v = y[1..].dot(&z[1..]);

    x[0] = (y[0] * z[0] - v) * pinv;

    let c1 = pinv * (v / y[0] - z[0]);
    let c2 = T::recip(y[0]);
    x[1..].waxpby(c1, &y[1..], c2, &z[1..]);
}

// ---------------------------------------------
// internal operations for second order cones
// ---------------------------------------------

fn _soc_residual<T>(z: &[T]) -> T
where
    T: FloatT,
{
    z[0] * z[0] - z[1..].sumsq()
}

fn _sqrt_soc_residual<T>(z: &[T]) -> T
where
    T: FloatT,
{
    let res = _soc_residual(z);
    if res > T::zero() {
        T::sqrt(res)
    } else {
        T::zero()
    }
}

// compute the residual at z + \alpha dz
// without storing the intermediate vector
fn _soc_residual_shifted<T>(z: &[T], dz: &[T], α: T) -> T
where
    T: FloatT,
{
    let x0 = z[0] + α * dz[0];
    let x1_sq = <[T] as VectorMath>::dot_shifted(&z[1..], &z[1..], &dz[1..], &dz[1..], α);

    x0 * x0 - x1_sq
}

// find the maximum step length α≥0 so that
// x + αy stays in the SOC
fn _step_length_soc_component<T>(x: &[T], y: &[T], αmax: T) -> T
where
    T: FloatT,
{
    // assume that x is in the SOC, and find the minimum positive root
    // of the quadratic equation:  ||x₁+αy₁||^2 = (x₀ + αy₀)^2

    let two: T = (2.).as_T();
    let four: T = (4.).as_T();

    let a = _soc_residual(y); //NB: could be negative
    let b = two * (x[0] * y[0] - x[1..].dot(&y[1..]));
    let c = T::max(T::zero(), _soc_residual(x)); //should be ≥0
    let d = b * b - four * a * c;

    if c < T::zero() {
        // This should never be reachable since c ≥ 0 above
        panic!("starting point of line search not in SOC");
    }

    #[allow(clippy::if_same_then_else)] // allows explanation of separate cases
    if (a > T::zero() && b > T::zero()) || d < T::zero() {
        //all negative roots / complex root pair
        //-> infinite step length
        return αmax;
    } else if a == T::zero() {
        // edge case with only one root.  This corresponds to
        // the case where the search direction is exactly on the
        // cone boundary.   The root should be -c/b, but b can't
        // be negative since both (x,y) are in the cone and it is
        // self dual, so <x,y> \ge 0 necessarily.
        return αmax;
    } else if c == T::zero() {
        // Edge case with one of the roots at 0.   This corresponds
        // to the case where the initial point is exactly on the
        // cone boundary.  The other root is -b/a.   If the search
        // direction is in the cone, then a >= 0 and b can't be
        // negative due to self-duality.  If a < 0, then the
        // direction is outside the cone and b can't be positive.
        // Either way, step length is determined by whether or not
        // the search direction is in the cone.
        return if a >= T::zero() { αmax } else { T::zero() };
    }

    // if we got this far then we need to calculate a pair
    // of real roots and choose the smallest positive one.
    // We need to be cautious about cancellations though.
    // See §1.4: Goldberg, ACM Computing Surveys, 1991
    // https://dl.acm.org/doi/pdf/10.1145/103162.103163

    let t = {
        if b >= T::zero() {
            -b - T::sqrt(d)
        } else {
            -b + T::sqrt(d)
        }
    };

    let r1: T = (two * c) / t;
    let r2: T = t / (two * a);

    // return the minimum positive root, up to αmax
    let r1 = if r1 < T::zero() { T::infinity() } else { r1 };
    let r2 = if r2 < T::zero() { T::infinity() } else { r2 };

    T::min(αmax, T::min(r1, r2))
}

// Must move the actual implementations of W*x to an outside
// fcn.  The operation λ = Wz produces a borrow conflict
// otherwise because λ is part of the cone's internal data
// and we can't borrow self and &mut λ at the same time.
// Could also have been done using std::mem::take

#[allow(non_snake_case)]
fn _soc_mul_W_inner<T>(y: &mut [T], x: &[T], α: T, β: T, w: &[T], η: T)
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

fn _soc_mul_Winv_inner<T>(y: &mut [T], x: &[T], α: T, β: T, w: &[T], η: T)
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
