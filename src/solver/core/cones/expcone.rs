use super::*;
use crate::{
    algebra::*,
    solver::{core::ScalingStrategy, CoreSettings},
};

// -------------------------------------
// Exponential Cone
// -------------------------------------

pub struct ExponentialCone<T> {
    // Hessian of the dual barrier at z
    H_dual: DenseMatrixSym3<T>,

    // scaling matrix, i.e. μH(z)
    Hs: DenseMatrixSym3<T>,

    // gradient of the dual barrier at z
    grad: [T; 3],

    // holds copy of z at scaling point
    z: [T; 3],
}

#[allow(clippy::new_without_default)]
impl<T> ExponentialCone<T>
where
    T: FloatT,
{
    pub fn new() -> Self {
        Self {
            H_dual: DenseMatrixSym3::zeros(),
            Hs: DenseMatrixSym3::zeros(),
            grad: [T::zero(); 3],
            z: [T::zero(); 3],
        }
    }
}

impl<T> Cone<T> for ExponentialCone<T>
where
    T: FloatT,
{
    fn degree(&self) -> usize {
        3
    }

    fn numel(&self) -> usize {
        3
    }

    fn is_symmetric(&self) -> bool {
        false
    }

    fn is_sparse_expandable(&self) -> bool {
        false
    }

    fn allows_primal_dual_scaling(&self) -> bool {
        true
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
        s[0] = (-1.051_383_945_322_714).as_T();
        s[1] = (0.556_409_619_469_370).as_T();
        s[2] = (1.258_967_884_768_947).as_T();

        (z[0], z[1], z[2]) = (s[0], s[1], s[2]);
    }

    fn set_identity_scaling(&mut self) {
        // We should never use identity scaling because
        // we never want to allow symmetric initialization
        unreachable!();
    }

    fn update_scaling(
        &mut self,
        s: &[T],
        z: &[T],
        μ: T,
        scaling_strategy: ScalingStrategy,
    ) -> bool {
        // update both gradient and Hessian for function f*(z) at the point z
        self.update_dual_grad_H(z);

        // update the scaling matrix Hs
        self.update_Hs(s, z, μ, scaling_strategy);

        // K.z .= z
        self.z.copy_from(z);

        true
    }

    fn Hs_is_diagonal(&self) -> bool {
        false
    }

    fn get_Hs(&self, Hsblock: &mut [T]) {
        // Hs data is already in packed triu form, so just copy
        Hsblock.copy_from(&self.Hs.data);
    }

    fn mul_Hs(&mut self, y: &mut [T], x: &[T], _work: &mut [T]) {
        self.Hs.mul(y, x);
    }

    fn affine_ds(&self, ds: &mut [T], s: &[T]) {
        ds.copy_from(s);
    }

    fn combined_ds_shift(&mut self, shift: &mut [T], step_z: &mut [T], step_s: &mut [T], σμ: T) {
        //3rd order correction requires input variables.z

        let mut η = [T::zero(); 3];
        self.higher_correction(&mut η, step_s, step_z);

        for i in 0..3 {
            shift[i] = self.grad[i] * σμ - η[i];
        }
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
        let mut work = [T::zero(); 3];

        let _is_prim_feasible_fcn = |s: &[T]| -> bool { self.is_primal_feasible(s) };
        let _is_dual_feasible_fcn = |s: &[T]| -> bool { self.is_dual_feasible(s) };

        let αz = backtrack_search(dz, z, αmax, αmin, step, _is_dual_feasible_fcn, &mut work);
        let αs = backtrack_search(ds, s, αmax, αmin, step, _is_prim_feasible_fcn, &mut work);

        (αz, αs)
    }

    fn compute_barrier(&mut self, z: &[T], s: &[T], dz: &[T], ds: &[T], α: T) -> T {
        let mut barrier = T::zero();

        let cur_z = [z[0] + α * dz[0], z[1] + α * dz[1], z[2] + α * dz[2]];
        let cur_s = [s[0] + α * ds[0], s[1] + α * ds[1], s[2] + α * ds[2]];

        barrier += self.barrier_dual(&cur_z);
        barrier += self.barrier_primal(&cur_s);

        barrier
    }
}

impl<T> NonsymmetricCone<T> for ExponentialCone<T>
where
    T: FloatT,
{
    // -----------------------------------------
    // internal operations for exponential cones
    //
    // Primal exponential cone: s3 ≥ s2*e^(s1/s2), s3,s2 > 0
    // Dual exponential cone: z3 ≥ -z1*e^(z2/z1 - 1), z3 > 0, z1 < 0
    // ----------------------------------------

    // Returns true if s is primal feasible
    fn is_primal_feasible(&self, s: &[T]) -> bool
    where
        T: FloatT,
    {
        if s[2] > T::zero() && s[1] > T::zero() {
            //feasible
            let res = s[1] * (s[2] / s[1]).logsafe() - s[0];
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
        if z[2] > T::zero() && z[0] < T::zero() {
            let res = z[1] - z[0] - z[0] * (-z[2] / z[0]).logsafe();
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
        // Primal barrier:
        // f(s) = ⟨s,g(s)⟩ - f*(-g(s))
        //      = -2*log(s2) - log(s3) - log((1-barω)^2/barω) - 3,
        // where barω = ω(1 - s1/s2 - log(s2) - log(s3))
        // NB: ⟨s,g(s)⟩ = -3 = - ν

        let ω = _wright_omega(T::one() - s[0] / s[1] - (s[1] / s[2]).logsafe());

        let ω = (ω - T::one()) * (ω - T::one()) / ω;

        -ω.logsafe() - (s[1].logsafe()) * ((2.).as_T()) - s[2].logsafe() - (3.).as_T()
    }

    fn barrier_dual(&mut self, z: &[T]) -> T
    where
        T: FloatT,
    {
        // Dual barrier:
        // f*(z) = -log(z2 - z1 - z1*log(z3/-z1)) - log(-z1) - log(z3)
        // -----------------------------------------
        let l = (-z[2] / z[0]).logsafe();
        -(-z[2] * z[0]).logsafe() - (z[1] - z[0] - z[0] * l).logsafe()
    }

    fn higher_correction(&mut self, η: &mut [T], ds: &[T], v: &[T])
    where
        T: FloatT,
    {
        // u for H^{-1}*Δs
        let H = &self.H_dual;
        let mut u = [T::zero(); 3];
        let z = &self.z;

        //Fine to use symmetric here because the upper
        //triangle is ignored anyway
        let mut cholH = DenseMatrixSym3::zeros();

        // solve H*u = ds
        let issuccess = cholH.cholesky_3x3_explicit_factor(H);
        if issuccess {
            cholH.cholesky_3x3_explicit_solve(&mut u[..], ds);
        } else {
            η.set(T::zero());
            return;
        }

        η[1] = T::one();
        η[2] = -z[0] / z[2]; // gradient of ψ
        η[0] = η[2].logsafe();

        let ψ = z[0] * η[0] - z[0] + z[1];

        let dotψu = u.dot(η);
        let dotψv = v.dot(η);

        let two: T = (2.).as_T();
        let coef =
            ((u[0] * (v[0] / z[0] - v[2] / z[2]) + u[2] * (z[0] * v[2] / z[2] - v[0]) / z[2]) * ψ
                - two * dotψu * dotψv)
                / (ψ * ψ * ψ);

        η.scale(coef);

        let inv_ψ2 = (ψ * ψ).recip();

        // efficient implementation for η above
        η[0] += (ψ.recip() - two / z[0]) * u[0] * v[0] / (z[0] * z[0])
            - u[2] * v[2] / (z[2] * z[2]) / ψ
            + dotψu * inv_ψ2 * (v[0] / z[0] - v[2] / z[2])
            + dotψv * inv_ψ2 * (u[0] / z[0] - u[2] / z[2]);
        η[2] += two * (z[0] / ψ - T::one()) * u[2] * v[2] / (z[2] * z[2] * z[2])
            - (u[2] * v[0] + u[0] * v[2]) / (z[2] * z[2]) / ψ
            + dotψu * inv_ψ2 * (z[0] * v[2] / (z[2] * z[2]) - v[0] / z[2])
            + dotψv * inv_ψ2 * (z[0] * u[2] / (z[2] * z[2]) - u[0] / z[2]);

        η[..].scale((0.5).as_T());
    }

    // 3rd-order correction at the point z.  Output is η.
    //
    // η = -0.5*[(dot(u,Hψ,v)*ψ - 2*dotψu*dotψv)/(ψ*ψ*ψ)*gψ +
    //      dotψu/(ψ*ψ)*Hψv + dotψv/(ψ*ψ)*Hψu - dotψuv/ψ + dothuv]
    //
    // where :
    // Hψ = [  1/z[1]    0   -1/z[3];
    //           0       0   0;
    //         -1/z[3]   0   z[1]/(z[3]*z[3]);]
    // dotψuv = [-u[1]*v[1]/(z[1]*z[1]) + u[3]*v[3]/(z[3]*z[3]);
    //            0;
    //           (u[3]*v[1]+u[1]*v[3])/(z[3]*z[3]) - 2*z[1]*u[3]*v[3]/(z[3]*z[3]*z[3])]
    //
    // dothuv = [-2*u[1]*v[1]/(z[1]*z[1]*z[1]) ;
    //            0;
    //           -2*u[3]*v[3]/(z[3]*z[3]*z[3])]
    // Hψv = Hψ*v
    // Hψu = Hψ*u
    // gψ is used inside η

    fn update_dual_grad_H(&mut self, z: &[T]) {
        let grad = &mut self.grad;
        let H = &mut self.H_dual;

        // Hessian computation, compute μ locally
        let l = (-z[2] / z[0]).logsafe();
        let r = -z[0] * l - z[0] + z[1];

        // compute the gradient at z
        let c2 = r.recip();

        grad[0] = c2 * l - z[0].recip();
        grad[1] = -c2;
        grad[2] = (c2 * z[0] - T::one()) / z[2];

        // compute_Hessian(K,z,H).   Type is symmetric, so
        // only need to assign upper triangle.
        H[(0, 0)] = (r * r - z[0] * r + l * l * z[0] * z[0]) / (r * z[0] * z[0] * r);
        H[(0, 1)] = -l / (r * r);
        H[(1, 1)] = (r * r).recip();
        H[(0, 2)] = (z[1] - z[0]) / (r * r * z[2]);
        H[(1, 2)] = -z[0] / (r * r * z[2]);
        H[(2, 2)] = (r * r - z[0] * r + z[0] * z[0]) / (r * r * z[2] * z[2]);
    }
}

impl<T> Nonsymmetric3DCone<T> for ExponentialCone<T>
where
    T: FloatT,
{
    // Compute the primal gradient of f(s) at s
    fn gradient_primal(&self, s: &[T]) -> [T; 3]
    where
        T: FloatT,
    {
        let mut g = [T::zero(); 3];
        let ω = _wright_omega(T::one() - s[0] / s[1] - (s[1] / s[2]).logsafe());

        g[0] = T::one() / ((ω - T::one()) * s[1]);
        g[1] = g[0] + g[0] * ((ω * s[1] / s[2]).logsafe()) - T::one() / s[1];
        g[2] = ω / ((T::one() - ω) * s[2]);
        g
    }

    //getters
    fn split_borrow_mut(
        &mut self,
    ) -> (
        &mut DenseMatrixSym3<T>,
        &mut DenseMatrixSym3<T>,
        &mut [T; 3],
        &mut [T; 3],
    ) {
        (&mut self.H_dual, &mut self.Hs, &mut self.grad, &mut self.z)
    }
}

// ω(z) is the Wright-Omega function
// Computes the value ω(z) defined as the solution y to
// y+log(y) = z for reals z>=1.
//
// Follows Algorithm 4, §8.4 of thesis of Santiago Serrango:
//  Algorithms for Unsymmetric Cone Optimization and an
//  Implementation for Problems with the Exponential Cone
//  https://web.stanford.edu/group/SOL/dissertations/ThesisAkleAdobe-augmented.pdf

fn _wright_omega<T>(z: T) -> T
where
    T: FloatT,
{
    if z < T::zero() {
        panic!("argument not in supported range");
    }

    let mut p: T;
    let mut w: T;
    if z < T::one() + T::PI() {
        //Initialize with the taylor series
        let zm1 = z - T::one();
        p = zm1; //(z-1)
        w = T::one() + p * ((0.5).as_T());
        p *= zm1; //(z-1)^2
        w += p * (1. / 16.0).as_T();
        p *= zm1; //(z-1)^3
        w -= p * (1. / 192.0).as_T();
        p *= zm1; //(z-1)^4
        w -= p * (1. / 3072.0).as_T();
        p *= zm1; //(z-1)^5
        w += p * (13. / 61440.0).as_T();
    } else {
        // Initialize with:
        // w(z) = z - log(z) +
        //        log(z)/z +
        //        log(z)/z^2(log(z)/2-1) +
        //        log(z)/z^3(1/3log(z)^2-3/2log(z)+1)

        let logz = z.logsafe();
        let zinv = z.recip();
        w = z - logz;

        // add log(z)/z
        let mut q = logz * zinv; // log(z)/z
        w += q;

        // add log(z)/z^2(log(z)/2-1)
        q *= zinv; // log(z)/(z^2)
        w += q * (logz / (2.).as_T() - T::one());

        // add log(z)/z^3(1/3log(z)^2-3/2log(z)+1)
        q *= zinv; // log(z)/(z^3)
        w += q * (logz * logz / (3.).as_T() - logz * (1.5).as_T() + T::one());
    }

    // Initialize the residual
    let mut r = z - w - w.logsafe();

    // Santiago suggests two refinement iterations only
    for _ in 0..2 {
        let wp1 = w + T::one();
        let t = wp1 * (wp1 + (r * (2.).as_T()) / (3.0).as_T());
        w *= T::one() + (r / wp1) * (t - r * (0.5).as_T()) / (t - r);

        let r_4th = r * r * r * r;
        let wp1_6th = wp1 * wp1 * wp1 * wp1 * wp1 * wp1;
        r = (w * w * (2.).as_T() - w * (8.).as_T() - T::one()) / (wp1_6th * (72.0).as_T()) * r_4th;
    }

    w
}

// internal unit tests
#[test]
fn test_wright_omega() {
    // y = ω(z) should solve y + ln(y) = z.
    let pts = [1e-7, 1e-5, 1e-3, 1e-1, 1e1, 1e3, 1e5, 1e7, 1e9];

    for z in pts {
        let y = _wright_omega(z);
        let zsolved = y + f64::ln(y);
        let err = f64::abs(z - zsolved);
        assert!((err / z) < 1e-9);
    }
}
