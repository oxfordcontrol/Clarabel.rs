use super::*;
use crate::{
    algebra::*,
    solver::{core::ScalingStrategy, CoreSettings},
};

// -------------------------------------
// Generalized Power Cone
// -------------------------------------

pub struct GenPowerCone<T: FloatT = f64> {
    // power defining the cone
    α: Vec<T>,
    // gradient of the dual barrier at z
    grad: Vec<T>,
    // holds copy of z at scaling point
    z: Vec<T>,
    // dimension of u, w and their sum
    pub dim1: usize,
    pub dim2: usize,
    dim: usize,
    // central path parameter
    pub μ: T,
    // vectors for rank 3 update representation of Hs
    pub p: Vec<T>,
    pub q: Vec<T>,
    pub r: Vec<T>,
    // first part of the diagonal
    d1: Vec<T>,
    // additional scalar terms for rank-2 rep
    d2: T,
    // additional constant for initialization in the Newton-Raphson method
    ψ: T,
}

impl<T> GenPowerCone<T>
where
    T: FloatT,
{
    pub fn new(α: Vec<T>, dim1: usize, dim2: usize) -> Self {
        let dim = dim1 + dim2;

        //PJG : this check belongs elsewhere
        assert!(α.iter().all(|r| *r > T::zero())); // check all powers are greater than 0

        // YC: should polish this check
        let mut powerr = T::zero();
        α.iter().for_each(|x| powerr += *x);
        powerr -= T::one();
        assert!(powerr.abs() < T::epsilon()); //check the sum of powers is 1

        let ψ = T::one() / (α.sumsq());

        Self {
            α,
            grad: vec![T::zero(); dim],
            z: vec![T::zero(); dim],
            dim1,
            dim2,
            dim,
            μ: T::one(),
            p: vec![T::zero(); dim],
            q: vec![T::zero(); dim1],
            r: vec![T::zero(); dim2],
            d1: vec![T::zero(); dim1],
            d2: T::zero(),
            ψ,
        }
    }
}

impl<T> Cone<T> for GenPowerCone<T>
where
    T: FloatT,
{
    fn degree(&self) -> usize {
        self.dim1 + 1
    }

    fn numel(&self) -> usize {
        self.dim
    }

    fn is_symmetric(&self) -> bool {
        false
    }

    fn rectify_equilibration(&self, δ: &mut [T], e: &[T]) -> bool {
        δ.copy_from(e).recip().scale(e.mean());
        true // scalar equilibration
    }

    fn margins(&mut self, _z: &mut [T], _pd: PrimalOrDualCone) -> (T, T) {
        // We should never end up shifting to this cone, since
        // asymmetric problems should always use unit_initialization
        unreachable!();
    }
    fn scaled_unit_shift(&self, _z: &mut [T], _α: T, _pd: PrimalOrDualCone) {
        // We should never end up shifting to this cone, since
        // asymmetric problems should always use unit_initialization
        unreachable!();
    }

    fn unit_initialization(&self, z: &mut [T], s: &mut [T]) {
        let α = &self.α;

        s[..α.len()]
            .iter_mut()
            .enumerate()
            .for_each(|(i, s)| *s = (T::one() + α[i]).sqrt());
        s[α.len()..].iter_mut().for_each(|x| *x = T::zero());

        z.copy_from(&s);
    }

    fn set_identity_scaling(&mut self) {
        // We should never use identity scaling because
        // we never want to allow symmetric initialization
        unreachable!();
    }

    fn update_scaling(
        &mut self,
        _s: &[T],
        z: &[T],
        μ: T,
        _scaling_strategy: ScalingStrategy,
    ) -> bool {
        // update both gradient and Hessian for function f*(z) at the point z
        self.update_dual_grad_H(z);
        self.μ = μ;

        // K.z .= z
        self.z.copy_from(z);

        true
    }

    fn Hs_is_diagonal(&self) -> bool {
        true
    }

    fn get_Hs(&self, Hsblock: &mut [T]) {
        // we are returning here the diagonal D = [d1; d2] block
        let d1 = &self.d1;
        Hsblock[..self.dim1]
            .iter_mut()
            .enumerate()
            .for_each(|(i, x)| *x = self.μ * d1[i]);
        let c = self.μ * self.d2;
        Hsblock[self.dim1..].iter_mut().for_each(|x| *x = c);
    }

    fn mul_Hs(&mut self, y: &mut [T], x: &[T], _work: &mut [T]) {
        let d1 = &self.d1;
        let d2 = self.d2;
        let dim1 = self.dim1;

        let coef_p = self.p.dot(x);
        let coef_q = self.q.dot(&x[..dim1]);
        let coef_r = self.r.dot(&x[dim1..]);

        let x1 = &x[..dim1];
        let x2 = &x[dim1..];

        y.iter_mut()
            .enumerate()
            .for_each(|(i, x)| *x = coef_p * self.p[i]);
        y[..dim1]
            .iter_mut()
            .enumerate()
            .for_each(|(i, x)| *x += d1[i] * x1[i] - coef_q * self.q[i]);
        y[dim1..]
            .iter_mut()
            .enumerate()
            .skip(dim1)
            .for_each(|(i, x)| *x += d2 * x2[i] - coef_r * self.r[i]);
        y.iter_mut().for_each(|x| *x *= self.μ);
    }

    fn affine_ds(&self, ds: &mut [T], s: &[T]) {
        ds.copy_from(s);
    }

    fn combined_ds_shift(
        &mut self, shift: &mut [T], _step_z: &mut [T], _step_s: &mut [T], σμ: T
    ) {
        //YC: No 3rd order correction at present

        let grad = &self.grad;
        shift
            .iter_mut()
            .enumerate()
            .for_each(|(i, x)| *x = σμ * grad[i]);
    }

    fn Δs_from_Δz_offset(&mut self, out: &mut [T], ds: &[T], _work: &mut [T], _z: &[T]) {
        out.copy_from(ds);
    }

    fn step_length(
        &mut self,
        dz: &[T],
        ds: &[T],
        z: &[T],
        s: &[T],
        settings: &CoreSettings<T>,
        αmax: T,
    ) -> (T, T) {
        let step = settings.linesearch_backtrack_step;
        let αmin = settings.min_terminate_step_length;

        // final backtracked position

        let _is_prim_feasible_fcn = |s: &[T]| -> bool { self.is_primal_feasible(s) };
        let _is_dual_feasible_fcn = |s: &[T]| -> bool { self.is_dual_feasible(s) };

        //YC: Do we need to care about the memory allocation time for wq?
        let mut wq = Vec::with_capacity(self.dim);

        let αz = self.step_length_n_cone(&mut wq, dz, z, αmax, αmin, step, _is_dual_feasible_fcn);
        let αs = self.step_length_n_cone(&mut wq, ds, s, αmax, αmin, step, _is_prim_feasible_fcn);

        (αz, αs)
    }

    fn compute_barrier(&self, z: &[T], s: &[T], dz: &[T], ds: &[T], α: T) -> T {
        let mut barrier = T::zero();

        //YC: Do we need to care about the memory allocation time for wq?
        let mut wq = Vec::with_capacity(self.dim);

        wq.iter_mut()
            .enumerate()
            .for_each(|(i, x)| *x = z[i] + α * dz[i]);
        barrier += self.barrier_dual(&wq);

        wq.iter_mut()
            .enumerate()
            .for_each(|(i, x)| *x = s[i] + α * ds[i]);
        barrier += self.barrier_primal(&wq);

        barrier
    }
}

//-------------------------------------
// Dual scaling
//-------------------------------------

//YC: better to define an additional trait "NonsymmetricCone" for it
impl<T> NonsymmetricCone<T> for GenPowerCone<T>
where
    T: FloatT,
{
    // Returns true if s is primal feasible
    fn is_primal_feasible(&self, s: &[T]) -> bool
    where
        T: FloatT,
    {
        let α = &self.α;
        let two: T = (2f64).as_T();
        let dim1 = self.dim1;

        if s[..dim1].iter().all(|x| *x > T::zero()) {
            let mut res = T::zero();
            for i in 0..dim1 {
                res += two * α[i] * s[i].logsafe()
            }
            res = T::exp(res) - s[dim1..].sumsq();

            if res > T::zero() {
                return true;
            }
        }
        false
    }

    // Returns true if z is dual feasible
    fn is_dual_feasible(&self, z: &[T]) -> bool
    where
        T: FloatT,
    {
        let α = &self.α;
        let two: T = (2.).as_T();
        let dim1 = self.dim1;

        if z[..dim1].iter().all(|x| *x > T::zero()) {
            let mut res = T::zero();
            for i in 0..dim1 {
                res += two * α[i] * (z[i] / α[i]).logsafe()
            }
            res = T::exp(res) - z[dim1..].sumsq();

            if res > T::zero() {
                return true;
            }
        }
        false
    }

    // Compute the primal gradient of f(s) at s
    fn minus_gradient_primal(&self, s: &[T]) -> (T, T)
    where
        T: FloatT,
    {
        let α = &self.α;
        let dim1 = self.dim1;
        let two: T = (2.).as_T();

        // unscaled phi
        let mut phi = T::one();
        for i in 0..dim1 {
            phi *= s[i].powf(two * α[i]);
        }

        // obtain g1 from the Newton-Raphson method
        let norm_r = s[dim1..].norm();
        let mut g1 = T::zero();

        if norm_r > T::epsilon() {
            g1 = _newton_raphson_genpowcone(norm_r, &s[..dim1], phi, α, self.ψ);
            // minus_g.iter_mut().enumerate().skip(dim1).for_each(|(i,x)| *x = g1*s[i]/norm_r);
            // minus_g[..dim1].iter_mut().enumerate().for_each(|(i,x)| *x = -(T::one()+α[i]+α[i]*g1*norm_r)/s[i]);
        }

        (g1, norm_r)
    }

    fn update_dual_grad_H(&mut self, z: &[T]) {
        let α = &self.α;
        let dim1 = self.dim1;
        let two: T = (2.).as_T();

        let mut phi = T::one();
        for i in 0..dim1 {
            phi *= (z[0] / α[i]).powf(two * α[i])
        }
        let norm2w = z[dim1..].sumsq();
        let ζ = phi - norm2w;
        assert!(ζ > T::zero());

        // compute the gradient at z
        let grad = &mut self.grad;
        let τ = &mut self.q;
        τ.iter_mut()
            .enumerate()
            .for_each(|(i, τ)| *τ = two * α[i] / z[i]);
        grad[..dim1]
            .iter_mut()
            .enumerate()
            .for_each(|(i, grad)| *grad = -τ[i] * phi / ζ - (T::one() - α[i]) / z[i]);
        grad.iter_mut()
            .enumerate()
            .skip(dim1)
            .for_each(|(i, x)| *x = two * z[i] / ζ);

        // compute Hessian information at z
        let p0 = (phi * (phi + norm2w) / two).sqrt();
        let p1 = -two * phi / p0;
        let q0 = (ζ * phi / two).sqrt();
        let r1 = two * (ζ / (phi + norm2w)).sqrt();

        // compute the diagonal d1,d2
        let d1 = &mut self.d1;
        d1.iter_mut()
            .enumerate()
            .for_each(|(i, d1)| *d1 = τ[i] * phi / (ζ * z[i]) + (T::one() - α[i]) / (z[i] * z[i]));
        self.d2 = two / ζ;

        // compute p, q, r where τ shares memory with q
        let c1 = p0 / ζ;
        let p = &mut self.p;
        p[..dim1].copy_from(&τ);
        p[..dim1].iter_mut().for_each(|x| *x *= c1);
        let c2 = p1 / ζ;
        p[dim1..].copy_from(&z[self.dim1..]);
        p[dim1..].iter_mut().for_each(|x| *x *= c2);

        let c3 = q0 / ζ;
        let q = &mut self.q;
        q.iter_mut().for_each(|x| *x *= c3);
        let c4 = r1 / ζ;
        let r = &mut self.r;
        r.copy_from(&z[dim1..]);
        r.iter_mut().for_each(|x| *x *= c4);
    }

    fn barrier_dual(&self, z: &[T]) -> T
    where
        T: FloatT,
    {
        // Dual barrier:
        let α = &self.α;
        let dim1 = self.dim1;
        let two: T = (2.).as_T();
        let mut res = T::zero();

        for i in 0..dim1 {
            res += two * α[i] * (z[i] / α[i]).logsafe();
        }
        res = T::exp(res) - z[dim1..].sumsq();

        let mut barrier: T = -res.logsafe();
        z[..dim1]
            .iter()
            .enumerate()
            .for_each(|(i, x)| barrier -= (*x).logsafe() * (T::one() - α[i]));

        barrier
    }

    fn barrier_primal(&self, s: &[T]) -> T
    where
        T: FloatT,
    {
        // Primal barrier: f(s) = ⟨s,g(s)⟩ - f*(-g(s))
        // NB: ⟨s,g(s)⟩ = -(dim1+1) = - ν
        let α = &self.α;

        //YC: Do we need to care about the memory allocation time for minus_q?
        let (g1, norm_r) = self.minus_gradient_primal(s);
        let mut minus_g = Vec::with_capacity(self.dim);

        let dim1 = self.dim1;
        if norm_r > T::epsilon() {
            minus_g
                .iter_mut()
                .enumerate()
                .skip(dim1)
                .for_each(|(i, x)| *x = g1 * s[i] / norm_r);
            minus_g[..dim1]
                .iter_mut()
                .enumerate()
                .for_each(|(i, x)| *x = -(T::one() + α[i] + α[i] * g1 * norm_r) / s[i]);
        } else {
            minus_g.iter_mut().skip(dim1).for_each(|x| *x = T::zero());
            minus_g[..dim1]
                .iter_mut()
                .enumerate()
                .for_each(|(i, x)| *x = -(T::one() + α[i]) / s[i]);
        }

        let minus_one = (-1.).as_T();
        minus_g.iter_mut().for_each(|x| *x *= minus_one); // add the sign to it, i.e. return -g

        let out = -self.barrier_dual(&minus_g) - self.degree().as_T();

        out
    }
}

// ----------------------------------------------
//  internal operations for generalized power cones

// Newton-Raphson method:
// solve a one-dimensional equation f(x) = 0
// x(k+1) = x(k) - f(x(k))/f'(x(k))
// When we initialize x0 such that 0 < x0 < x* and f(x0) > 0,
// the Newton-Raphson method converges quadratically

fn _newton_raphson_genpowcone<T>(norm_r: T, p: &[T], phi: T, α: &Vec<T>, ψ: T) -> T
where
    T: FloatT,
{
    let two: T = (2.).as_T();

    // init point x0: f(x0) > 0
    let x0 = -norm_r.recip()
        + (ψ * norm_r + ((phi / norm_r / norm_r + ψ * ψ - T::one()) * phi).sqrt())
            / (phi - norm_r * norm_r);

    // function for f(x) = 0
    let f0 = {
        |x: T| -> T {
            let mut fval = -(two * x / norm_r + x * x).logsafe();
            α.iter().enumerate().for_each(|(i, &αi)| {
                fval += two * αi * ((x * norm_r + (T::one() + αi) / αi).logsafe() - p[i].logsafe())
            });

            fval
        }
    };

    // first derivative
    let f1 = {
        |x: T| -> T {
            let mut fval = -(two * x + two / norm_r) / (x * x + two * x / norm_r);
            α.iter()
                .for_each(|&αi| fval += two * (αi) * norm_r / (norm_r * x + (T::one() + αi) / αi));

            fval
        }
    };
    _newton_raphson_onesided(x0, f0, f1)
}

// YC: it is the duplicate of the same one for power cones. Can we unify them into a common one?
fn _newton_raphson_onesided<T>(x0: T, f0: impl Fn(T) -> T, f1: impl Fn(T) -> T) -> T
where
    T: FloatT,
{
    // implements NR method from a starting point assumed to be to the
    // left of the true value.   Once a negative step is encountered
    // this function will halt regardless of the calculated correction.

    let mut x = x0;
    let mut iter = 0;

    while iter < 100 {
        iter += 1;
        let dfdx = f1(x);
        let dx = -f0(x) / dfdx;

        if (dx < T::epsilon())
            || (T::abs(dx / x) < T::sqrt(T::epsilon()))
            || (T::abs(dfdx) < T::epsilon())
        {
            break;
        }
        x += dx;
    }

    x
}
