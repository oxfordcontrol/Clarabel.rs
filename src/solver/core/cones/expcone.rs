use super::{Cone, Nonsymmetric3DCone, Nonsymmetric3DConeUtils};
use crate::{
    algebra::*,
    solver::{core::ScalingStrategy, CoreSettings},
};

// -------------------------------------
// Exponential Cone
// -------------------------------------

pub struct ExponentialCone<T: FloatT = f64> {
    // current μH and gradient
    H: DenseMatrix3<T>,
    grad: [T; 3],

    // workspace data
    HBFGS: DenseMatrix3<T>,
    z: [T; 3],

    cholH: DenseMatrix3<T>,
}

impl<T> ExponentialCone<T>
where
    T: FloatT,
{
    pub fn new() -> Self {
        Self {
            H: DenseMatrix3::zeros(),
            grad: [T::zero(); 3],

            // workspace data
            HBFGS: DenseMatrix3::zeros(),
            z: [T::zero(); 3],

            //cholesky factor of H
            cholH: DenseMatrix3::zeros(),
        }
    }
}

impl<T> Cone<T> for ExponentialCone<T>
where
    T: FloatT,
{
    fn dim(&self) -> usize {
        3
    }

    fn degree(&self) -> usize {
        self.dim()
    }

    fn numel(&self) -> usize {
        self.dim()
    }

    fn is_symmetric(&self) -> bool {
        false
    }

    fn rectify_equilibration(&self, δ: &mut [T], e: &[T]) -> bool {
        δ.copy_from(e);
        δ.reciprocal();
        δ.scale(e.mean());
        true // scalar equilibration
    }

    fn shift_to_cone(&self, _z: &mut [T]) {
        // We should never end up shifting to this cone, since
        // asymmetric problems should always use unit_initialization
        unimplemented!("This function should never be reached.");
    }

    fn unit_initialization(&self, z: &mut [T], s: &mut [T]) {
        s[0] = (-1.051383945322714).as_T();
        s[1] = (0.556409619469370).as_T();
        s[2] = (1.258967884768947).as_T();

        (z[0], z[1], z[2]) = (s[0], s[1], s[2]);
    }

    fn set_identity_scaling(&mut self) {
        // We should never use identity scaling because
        // we never want to allow symmetric initialization
        unimplemented!("This function should never be reached.");
    }

    fn update_scaling(&mut self, s: &[T], z: &[T], μ: T, scaling_strategy: ScalingStrategy) {
        // update both gradient and Hessian for function f*(z) at the point z
        // NB: the update order can't be switched as we reuse memory in the
        // Hessian computation Hessian update
        self.update_grad_HBFGS(s, z, μ, scaling_strategy);

        // K.z .= z
        self.z.copy_from(z);
    }

    fn Hs_is_diagonal(&self) -> bool {
        false
    }

    fn get_Hs(&self, Hsblock: &mut [T]) {
        // stores triu(K.HBFGS) into a vector
        self.HBFGS.pack_triu(Hsblock);
    }

    fn mul_Hs(&self, y: &mut [T], x: &[T], _work: &mut [T]) {
        let H = &self.HBFGS;
        for i in 0..3 {
            y[i] = H[(i, 0)] * x[0] + H[(i, 1)] * x[1] + H[(i, 2)] * x[2];
        }
    }

    fn affine_ds(&self, ds: &mut [T], s: &[T]) {
        ds.copy_from(s);
    }

    fn combined_ds_shift(&mut self, shift: &mut [T], step_z: &[T], step_s: &[T], σμ: T) {
        //3rd order correction requires input variables.z

        let mut η = [T::zero(); 3];
        self.higher_correction(&mut η, step_s, step_z);

        for i in 0..3 {
            shift[i] = self.grad[i] * σμ - η[i];
        }
    }

    fn Δs_from_Δz_offset(&self, out: &mut [T], ds: &[T], _work: &mut [T]) {
        out.copy_from(ds);
    }

    fn step_length(
        &self,
        dz: &[T],
        ds: &[T],
        z: &[T],
        s: &[T],
        settings: &CoreSettings<T>,
        αmax: T,
    ) -> (T, T) {
        let step = settings.linesearch_backtrack_step;
        let αmin = settings.min_terminate_step_length;

        let αz = self.step_length_3d_cone(dz, z, αmax, αmin, step, _is_dual_feasible_expcone);
        let αs = self.step_length_3d_cone(ds, s, αmax, αmin, step, _is_primal_feasible_expcone);

        (αz, αs)
    }

    fn compute_barrier(&self, z: &[T], s: &[T], dz: &[T], ds: &[T], α: T) -> T {
        let mut barrier = T::zero();

        let cur_z = [z[0] + α * dz[0], z[1] + α * dz[1], z[2] + α * dz[2]];
        let cur_s = [s[0] + α * ds[0], s[1] + α * ds[1], s[2] + α * ds[2]];

        barrier += _barrier_dual(&cur_z);
        barrier += _barrier_primal(&cur_s);

        barrier
    }
}

// implement this marker trait to get access to
// functions to common to exp and pow cones
impl<T> Nonsymmetric3DCone<T> for ExponentialCone<T> where T: FloatT {}

// -----------------------------------------
// internal operations for exponential cones
//
// Primal exponential cone: s3 ≥ s2*e^(s1/s2), s3,s2 > 0
// Dual exponential cone: z3 ≥ -z1*e^(z2/z1 - 1), z3 > 0, z1 < 0
// We use the dual barrier function:
// f*(z) = -log(z2 - z1 - z1*log(z3/-z1)) - log(-z1) - log(z3)
// -----------------------------------------

fn _barrier_dual<T>(z: &[T]) -> T
where
    T: FloatT,
{
    // Dual barrier
    let l = (-z[2] / z[0]).logsafe();
    -(-z[2] * z[0]).logsafe() - (z[1] - z[0] - z[0] * l).logsafe()
}

fn _barrier_primal<T>(s: &[T]) -> T
where
    T: FloatT,
{
    // Primal barrier:
    // f(s) = ⟨s,g(s)⟩ - f*(-g(s))
    //      = -2*log(s2) - log(s3) - log((1-barω)^2/barω) - 3,
    // where barω = ω(1 - s1/s2 - log(s2) - log(s3))
    // NB: ⟨s,g(s)⟩ = -3 = - ν

    //PJG: many compile errors.  Double check.

    let ω = _wright_omega(T::one() - s[0] / s[1] - (s[1] / s[2]).logsafe());

    let ω = (ω - T::one()) * (ω - T::one()) / ω;

    -ω.logsafe() - (s[1].logsafe()) * ((2.).as_T()) - s[2].logsafe() - (3.).as_T()
}

// Returns true if s is primal feasible
fn _is_primal_feasible_expcone<T>(s: &[T]) -> bool
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
fn _is_dual_feasible_expcone<T>(z: &[T]) -> bool
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

// Compute the primal gradient of f(s) at s
// solve it by the Newton-Raphson method
fn _gradient_primal<T>(g: &mut [T], s: &[T])
where
    T: FloatT,
{
    let ω = _wright_omega(T::one() - s[0] / s[1] - (s[1] / s[2]).logsafe());

    g[0] = T::one() / ((ω - T::one()) * s[1]);
    g[1] = g[0] + g[0] * ((ω * s[1] / s[2]).logsafe()) - T::one() / s[1];
    g[2] = ω / ((T::one() - ω) * s[2]);
}

// ω(z) is the Wright-Omega function
// Computes the value ω(z) defined as the solution y to
// y+log(y) = z for reals z>=1.
//
// Follows Algorithm 4, §8.4 of thesis of Santiago Serrango:
//  Algorithms for Unsymmetric Cone Optimization and an
//  Implementation for Problems with the Exponential Cone
//  https://web.stanford.edu/group/SOL/dissertations/ThesisAkleAdobe-augmented.pdf

//PJG: This function is crying out for a unit test,
//as are all of the other fiddly functions in this file
//I had to move a lot of constant terms around because
//I couldn't get two sided multiplication to work initially

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
    for _ in 0..3 {
        let wp1 = w + T::one();
        let t = wp1 * (wp1 + (r * (2.).as_T()) / (3.0).as_T());
        w *= T::one() + (r / wp1) * (t - r * (0.5).as_T()) / (t - r);

        let r_4th = r * r * r * r;
        let wp1_6th = wp1 * wp1 * wp1 * wp1 * wp1 * wp1;
        r = (w * w * (2.).as_T() - w * (8.).as_T() - T::one()) / (wp1_6th * (72.0).as_T()) * r_4th;
    }

    w
}

// 3rd-order correction at the point z,
// wrt directions u,v. Writes in η.

impl<T> ExponentialCone<T>
where
    T: FloatT,
{
    fn higher_correction(&mut self, η: &mut [T; 3], ds: &[T], v: &[T])
    where
        T: FloatT,
    {
        // u for H^{-1}*Δs
        let H = &self.H;
        let mut u = [T::zero(); 3];
        let z = &self.z;
        let cholH = &mut self.cholH;

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

        let dotψu = u.dot(&η[..]);
        let dotψv = v.dot(&η[..]);

        // 3rd order correction:
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
}
//-------------------------------------
// primal-dual scaling
//-------------------------------------

// Implementation sketch
// 1) only need to replace μH by W^TW, where
//    W^TW is the primal-dual scaling matrix
//    generated by BFGS, i.e. W^T W*[z,\tilde z] = [s,\tile s]
//   \tilde z = -f'(s), \tilde s = - f*'(z)

impl<T> ExponentialCone<T>
where
    T: FloatT,
{
    fn update_grad_HBFGS(&mut self, s: &[T], z: &[T], μ: T, scaling_strategy: ScalingStrategy) {
        let st = &mut self.grad;
        let mut zt = [T::zero(); 3];
        let mut δs = [T::zero(); 3];

        //shared for δz, tmp, axis_z
        let mut tmp = [T::zero(); 3];
        let H = &mut self.H;
        let HBFGS = &mut self.HBFGS;

        // Hessian computation, compute μ locally
        let l = (-z[2] / z[0]).logsafe();
        let r = -z[0] * l - z[0] + z[1];

        // compute the gradient at z
        let c2 = r.recip();

        st[0] = c2 * l - z[0].recip();
        st[1] = -c2;
        st[2] = (c2 * z[0] - T::one()) / z[2];

        // compute_Hessian(K,z,H)
        H[(0, 0)] = (r * r - z[0] * r + l * l * z[0] * z[0]) / (r * z[0] * z[0] * r);
        H[(0, 1)] = -l / (r * r);
        H[(1, 0)] = H[(0, 1)];
        H[(1, 1)] = (r * r).recip();
        H[(0, 2)] = (z[1] - z[0]) / (r * r * z[2]);
        H[(2, 0)] = H[(0, 2)];
        H[(1, 2)] = -z[0] / (r * r * z[2]);
        H[(2, 1)] = H[(1, 2)];
        H[(2, 2)] = (r * r - z[0] * r + z[0] * z[0]) / (r * r * z[2] * z[2]);

        // Use the local mu with primal dual strategy.  Otherwise
        // we use the global one
        if scaling_strategy == ScalingStrategy::Dual {
            //HBFGS .= μ*H
            for i in 0..3 {
                for j in 0..3 {
                    //PJG: not rusty
                    HBFGS[(i, j)] = μ * H[(i, j)];
                }
            }
        }

        let three: T = (3.).as_T();
        let dot_sz = s.dot(z);
        let μ = dot_sz / three;

        // compute zt,st,μt locally
        // NB: zt,st have different sign convention wrt Mosek paper
        _gradient_primal(&mut zt[..], s);

        let μt = st[..].dot(&zt[..]) / three;

        // δs = s + μ*st
        // δz = z + μ*zt
        let mut δz = tmp;

        for i in 0..3 {
            δs[i] = s[i] + μ * st[i];
            δz[i] = z[i] + μ * zt[i];
        }

        let dot_δsz = δs[..].dot(&δz[..]);

        let de1 = μ * μt - T::one();
        let de2 = H.quad_form(&zt, &zt) - three * μt * μt;

        if !(de1.abs() > T::epsilon() && de2.abs() > T::epsilon()) {
            // HBFGS when s,z are on the central path
            for i in 0..3 {
                for j in 0..3 {
                    //PJG: not rusty
                    HBFGS[(i, j)] = μ * H[(i, j)];
                }
            }
        } else {
            // compute t
            // tmp = μt*st - H*zt
            // PJG: need a gemv style function for DenseMatrix3
            for i in 0..3 {
                tmp[i] = μt * st[i] - H[(i, 0)] * zt[0] - H[(i, 1)] * zt[1] - H[(i, 2)] * zt[2];
            }

            // HBFGS as a workspace
            HBFGS.copy_from(H);
            for i in 0..3 {
                for j in i..3 {
                    HBFGS[(i, j)] -= st[i] * st[j] / three + tmp[i] * tmp[j] / de2;
                }
            }
            // symmetrize matrix
            HBFGS[(1, 0)] = HBFGS[(0, 1)];
            HBFGS[(2, 0)] = HBFGS[(0, 2)];
            HBFGS[(2, 1)] = HBFGS[(1, 2)];

            let t = μ * HBFGS.data.norm(); //Frobenius norm

            // generate the remaining axis
            // axis_z = cross(z,zt)
            let mut axis_z = tmp;
            axis_z[0] = z[1] * zt[2] - z[2] * zt[1];
            axis_z[1] = z[2] * zt[0] - z[0] * zt[2];
            axis_z[2] = z[0] * zt[1] - z[1] * zt[0];
            axis_z.normalize();

            // HBFGS = s*s'/⟨s,z⟩ + δs*δs'/⟨δs,δz⟩ + t*axis_z*axis_z'
            for i in 0..3 {
                for j in i..3 {
                    HBFGS[(i, j)] =
                        s[i] * s[j] / dot_sz + δs[i] * δs[j] / dot_δsz + t * axis_z[i] * axis_z[j];
                }
            }
            // symmetrize matrix
            HBFGS[(1, 0)] = HBFGS[(0, 1)];
            HBFGS[(2, 0)] = HBFGS[(0, 2)];
            HBFGS[(2, 1)] = HBFGS[(1, 2)];
        }
    }
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
