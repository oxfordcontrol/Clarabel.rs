use super::*;
use crate::algebra::*;
use itertools::izip;
use std::iter::zip;

// -------------------------------------
// Generalized Power Cone
// -------------------------------------

pub struct GenPowerConeData<T> {
    // gradient of the dual barrier at z
    grad: Vec<T>,
    // holds copy of z at scaling point
    z: Vec<T>,

    // central path parameter
    pub μ: T,

    // vectors for rank 3 update representation of Hs
    pub p: Vec<T>,
    pub q: Vec<T>,
    pub r: Vec<T>,
    pub d1: Vec<T>,

    // additional scalar terms for rank-2 rep
    d2: T,
    // additional constant for initialization in the Newton-Raphson method
    ψ: T,

    //work vector length dim, e.g. for line searches
    work: Vec<T>,
    //work vector exclusively for computing the primal barrier function.
    work_pb: Vec<T>,
}

impl<T> GenPowerConeData<T>
where
    T: FloatT,
{
    pub fn new(α: &[T], dim2: usize) -> Self {
        let dim1 = α.len();
        let dim = dim1 + dim2;

        //PJG : these checks belong elsewhere
        assert!(α.iter().all(|r| *r > T::zero())); // check all powers are greater than 0
        let len_t: T = α.len().as_T();
        assert!((T::one() - α.sum()).abs() < (T::epsilon() * len_t * (0.5).as_T()));

        Self {
            grad: vec![T::zero(); dim],
            z: vec![T::zero(); dim],
            μ: T::one(),
            p: vec![T::zero(); dim],
            q: vec![T::zero(); dim1],
            r: vec![T::zero(); dim2],
            d1: vec![T::zero(); dim1],
            d2: T::zero(),
            ψ: T::one() / (α.sumsq()),
            work: vec![T::zero(); dim],
            work_pb: vec![T::zero(); dim],
        }
    }
}

pub struct GenPowerCone<T> {
    pub α: Vec<T>,                      // power defining the cone.  length determines dim1
    dim2: usize,                        // dimension of w
    pub data: Box<GenPowerConeData<T>>, // Boxed so that the enum_dispatch variant isn't huge
}

impl<T> GenPowerCone<T>
where
    T: FloatT,
{
    pub fn new(α: Vec<T>, dim2: usize) -> Self {
        let data = Box::new(GenPowerConeData::<T>::new(&α, dim2));
        Self { α, dim2, data }
    }

    pub fn dim1(&self) -> usize {
        self.α.len()
    }
    pub fn dim2(&self) -> usize {
        self.dim2
    }
    pub fn dim(&self) -> usize {
        self.dim1() + self.dim2()
    }
}

impl<T> Cone<T> for GenPowerCone<T>
where
    T: FloatT,
{
    fn degree(&self) -> usize {
        self.dim1() + 1
    }

    fn numel(&self) -> usize {
        self.dim()
    }

    fn is_symmetric(&self) -> bool {
        false
    }

    fn is_sparse_expandable(&self) -> bool {
        // we do not curently have a way of representing
        // this cone in non-expanded form
        true
    }

    fn allows_primal_dual_scaling(&self) -> bool {
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
        let dim1 = self.dim1();

        s[..dim1].scalarop_from(|αi| T::sqrt(T::one() + αi), α);
        s[dim1..].set(T::zero());

        z.copy_from(s);
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
        self.data.μ = μ;

        // K.z .= z
        self.data.z.copy_from(z);

        true
    }

    fn Hs_is_diagonal(&self) -> bool {
        true
    }

    fn get_Hs(&self, Hsblock: &mut [T]) {
        // we are returning here the diagonal D = [d1; d2] block
        let dim1 = self.dim1();
        let data = &self.data;

        let μ_for_d1 = data.μ.clone();
        Hsblock[..dim1].scalarop_from(move |d1| μ_for_d1.clone() * d1, &data.d1);
        Hsblock[dim1..].set(data.μ.clone() * data.d2.clone());
    }

    fn mul_Hs(&mut self, y: &mut [T], x: &[T], _work: &mut [T]) {
        // Hs = μ*(D + pp' -qq' -rr')

        let dim1 = self.dim1();
        let data = &self.data;

        let coef_p = data.p.dot(x);
        let coef_q = data.q.dot(&x[..dim1]);
        let coef_r = data.r.dot(&x[dim1..]);

        // y1 .= K.d1 .* x1 - coef_q.*K.q
        // NB: d1 is a vector
        for (y, x, d1, q) in izip!(&mut y[..dim1], &x[..dim1], &data.d1, &data.q) {
            *y = d1.clone() * x.clone() - coef_q.clone() * q.clone();
        }

        // y2 .= K.d2 .* x2 - coef_r.*K.r.
        // NB: d2 is a scalar
        for (y, x, r) in izip!(&mut y[dim1..], &x[dim1..], &data.r) {
            *y = data.d2.clone() * x.clone() - coef_r.clone() * r.clone();
        }

        y.axpby(coef_p, &data.p, T::one());
        y.scale(data.μ.clone());
    }

    fn affine_ds(&self, ds: &mut [T], s: &[T]) {
        ds.copy_from(s);
    }

    fn combined_ds_shift(
        &mut self, shift: &mut [T], _step_z: &mut [T], _step_s: &mut [T], σμ: T
    ) {
        //YC: No 3rd order correction at present
        shift.scalarop_from(move |g| g * σμ.clone(), &self.data.grad);
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
        let step = settings.linesearch_backtrack_step.clone();
        let αmin = settings.min_terminate_step_length.clone();

        //simultaneously using "work" and the closures defined
        //below produces a borrow check error, so temporarily
        //move "work" out of self
        let mut work = std::mem::take(&mut self.data.work);

        let is_prim_feasible_fcn = |s: &[T]| -> bool { self.is_primal_feasible(s) };
        let is_dual_feasible_fcn = |s: &[T]| -> bool { self.is_dual_feasible(s) };

        let αz = backtrack_search(
            dz,
            z,
            αmax.clone(),
            αmin.clone(),
            step.clone(),
            is_dual_feasible_fcn,
            &mut work,
        );
        let αs = backtrack_search(ds, s, αmax, αmin, step, is_prim_feasible_fcn, &mut work);

        //restore work to self
        self.data.work = work;

        (αz, αs)
    }

    fn compute_barrier(&mut self, z: &[T], s: &[T], dz: &[T], ds: &[T], α: T) -> T {
        let mut barrier = T::zero();
        let mut work = std::mem::take(&mut self.data.work);

        work.waxpby(T::one(), s, α.clone(), ds);
        barrier += self.barrier_primal(&work);

        work.waxpby(T::one(), z, α, dz);
        barrier += self.barrier_dual(&work);

        self.data.work = work;

        barrier
    }
}

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
        let dim1 = self.dim1();

        if s[..dim1].iter().all(|x| *x > T::zero()) {
            let res = zip(α, &s[..dim1]).fold(T::zero(), |res, (αi, si)| -> T {
                res + two.clone() * αi.clone() * si.clone().logsafe()
            });
            let res = T::exp(res) - s[dim1..].sumsq();

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
        let dim1 = self.dim1();

        if z[..dim1].iter().all(|x| *x > T::zero()) {
            let res = zip(α, &z[..dim1]).fold(T::zero(), |res, (αi, zi)| -> T {
                res + two.clone() * αi.clone() * (zi.clone() / αi.clone()).logsafe()
            });
            let res = T::exp(res) - z[dim1..].sumsq();

            if res > T::zero() {
                return true;
            }
        }
        false
    }

    fn barrier_primal(&mut self, s: &[T]) -> T
    where
        T: FloatT,
    {
        // Primal barrier: f(s) = ⟨s,g(s)⟩ - f*(-g(s))
        // NB: ⟨s,g(s)⟩ = -(dim1(K)+1) = - ν

        // can't use "work" here because it was already
        // used to construct the argument s in some cases
        let mut g = std::mem::take(&mut self.data.work_pb);

        self.gradient_primal(&mut g, s);
        g.negate(); //-g(s)

        let degree_t: T = self.degree().as_T();
        let out = -self.barrier_dual(&g) - degree_t;

        self.data.work_pb = g;

        out
    }

    fn barrier_dual(&mut self, z: &[T]) -> T
    where
        T: FloatT,
    {
        // Dual barrier:
        let α = &self.α;
        let dim1 = self.dim1();
        let two: T = (2.).as_T();

        let mut res = T::zero();
        for (zi, αi) in zip(&z[..dim1], α) {
            res += two.clone() * αi.clone() * (zi.clone() / αi.clone()).logsafe();
        }
        res = T::exp(res) - z[dim1..].sumsq();

        let mut barrier: T = -res.logsafe();
        for (zi, αi) in zip(&z[..dim1], α) {
            barrier -= zi.clone().logsafe() * (T::one() - αi.clone());
        }

        barrier
    }

    fn higher_correction(&mut self, _η: &mut [T], _ds: &[T], _v: &[T]) {
        unimplemented!()
    }

    fn update_dual_grad_H(&mut self, z: &[T]) {
        let α = &self.α;
        let dim1 = self.dim1();
        let data = &mut self.data;
        let two: T = (2.).as_T();

        let phi = zip(α, z).fold(T::one(), |phi, (αi, zi)| {
            phi * (zi.clone() / αi.clone()).powf(two.clone() * αi.clone())
        });

        let norm2w = z[dim1..].sumsq();
        let ζ = phi.clone() - norm2w.clone();
        assert!(ζ > T::zero());

        // compute the gradient at z
        let grad = &mut data.grad;
        let τ = &mut data.q;

        for (τ, grad, α, z) in izip!(τ.iter_mut(), &mut grad[..dim1], α, &z[..dim1]) {
            *τ = two.clone() * α.clone() / z.clone();
            *grad = -(τ.clone()) * phi.clone() / ζ.clone() - (T::one() - α.clone()) / z.clone();
        }

        let two_over_ζ = two.clone() / ζ.clone();
        grad[dim1..].scalarop_from(move |z| two_over_ζ.clone() * z, &z[dim1..]);

        // compute Hessian information at z
        let p0 = T::sqrt(phi.clone() * (phi.clone() + norm2w.clone()) / two.clone());
        let p1 = -two.clone() * phi.clone() / p0.clone();
        let q0 = T::sqrt(ζ.clone() * phi.clone() / two.clone());
        let r1 = two.clone() * T::sqrt(ζ.clone() / (phi.clone() + norm2w));

        // compute the diagonal d1,d2
        for (d1, τ, α, z) in izip!(&mut data.d1, τ.iter(), α, &z[..dim1]) {
            *d1 = τ.clone() * phi.clone() / (ζ.clone() * z.clone())
                + (T::one() - α.clone()) / (z.clone() * z.clone());
        }
        data.d2 = two.clone() / ζ.clone();

        // compute p, q, r where τ shares memory with q
        let p0_over_ζ = p0 / ζ.clone();
        let p1_over_ζ = p1 / ζ.clone();
        let r1_over_ζ = r1 / ζ.clone();
        data.p[..dim1].scalarop_from(move |τi| p0_over_ζ.clone() * τi, τ);
        data.p[dim1..].scalarop_from(move |zi| p1_over_ζ.clone() * zi, &z[dim1..]);

        data.q.scale(q0 / ζ.clone());
        data.r.scalarop_from(move |zi| r1_over_ζ.clone() * zi, &z[dim1..]);
    }
}

impl<T> NonsymmetricNDCone<T> for GenPowerCone<T>
where
    T: FloatT,
{
    // Compute the primal gradient of f(s) at s
    fn gradient_primal(&self, g: &mut [T], s: &[T])
    where
        T: FloatT,
    {
        let dim1 = self.dim1();
        let two: T = (2.).as_T();
        let data = &self.data;

        // unscaled phi
        let phi = zip(&s[..dim1], &self.α).fold(T::one(), |phi, (si, αi)| {
            phi * si.clone().powf(two.clone() * αi.clone())
        });

        // obtain g1 from the Newton-Raphson method
        let (p, r) = s.split_at(dim1);
        let (gp, gr) = g.split_at_mut(dim1);
        let norm_r = r.norm();

        if norm_r > T::epsilon() {
            let g1 = _newton_raphson_genpowcone(norm_r.clone(), p, phi, &self.α, data.ψ.clone());

            let g1_over_norm_r = g1.clone() / norm_r.clone();
            gr.scalarop_from(move |r| g1_over_norm_r.clone() * r, r);

            for (gp, α, p) in izip!(gp.iter_mut(), &self.α, p) {
                *gp = -(T::one()
                    + α.clone()
                    + α.clone() * g1.clone() * norm_r.clone())
                    / p.clone();
            }
        } else {
            gr.set(T::zero());

            for (gp, α, p) in izip!(gp.iter_mut(), &self.α, p) {
                *gp = -(T::one() + α.clone()) / p.clone();
            }
        }
    }
}
// ----------------------------------------------
//  internal operations for generalized power cones

// Newton-Raphson method:
// solve a one-dimensional equation f(x) = 0
// x(k+1) = x(k) - f(x(k))/f'(x(k))
// When we initialize x0 such that 0 < x0 < x* and f(x0) > 0,
// the Newton-Raphson method converges quadratically

fn _newton_raphson_genpowcone<T>(norm_r: T, p: &[T], phi: T, α: &[T], ψ: T) -> T
where
    T: FloatT,
{
    let two: T = (2.).as_T();

    // init point x0: f(x0) > 0
    let x0 = -norm_r.clone().recip()
        + (ψ.clone() * norm_r.clone()
            + ((phi.clone() / norm_r.clone() / norm_r.clone() + ψ.clone() * ψ - T::one())
                * phi.clone())
            .sqrt())
            / (phi.clone() - norm_r.clone() * norm_r.clone());

    // function for f(x) = 0 (clones captured values per call)
    let two_f0 = two.clone();
    let norm_r_f0 = norm_r.clone();
    let f0 = move |x: T| -> T {
        let two = two_f0.clone();
        let norm_r = norm_r_f0.clone();
        let finit = -(two.clone() * x.clone() / norm_r.clone() + x.clone() * x.clone()).logsafe();

        zip(α, p).fold(finit, |f, (αi, pi)| {
            f + two.clone()
                * αi.clone()
                * ((x.clone() * norm_r.clone() + (T::one() + αi.clone()) / αi.clone()).logsafe()
                    - pi.clone().logsafe())
        })
    };

    // first derivative
    let two_f1 = two.clone();
    let norm_r_f1 = norm_r.clone();
    let f1 = move |x: T| -> T {
        let two = two_f1.clone();
        let norm_r = norm_r_f1.clone();
        let finit = -(two.clone() * x.clone() + two.clone() / norm_r.clone())
            / (x.clone() * x.clone() + two.clone() * x.clone() / norm_r.clone());

        α.iter().fold(finit, |f, αi| {
            f + two.clone() * αi.clone() * norm_r.clone()
                / (norm_r.clone() * x.clone() + (T::one() + αi.clone()) / αi.clone())
        })
    };
    newton_raphson_onesided(x0, f0, f1)
}
