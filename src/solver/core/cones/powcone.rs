use super::{Cone, Nonsymmetric3DCone, Nonsymmetric3DConeUtils};
use crate::{
    algebra::*,
    solver::{core::ScalingStrategy, CoreSettings},
};

// -------------------------------------
// Power Cone
// -------------------------------------

pub struct PowerCone<T: FloatT = f64> {
    // power defining the cone
    α: T,
    // Hessian of the dual barrier at z
    H_dual: DenseMatrixSym3<T>,

    // scaling matrix, i.e. μH(z)
    Hs: DenseMatrixSym3<T>,

    // gradient of the dual barrier at z
    grad: [T; 3],

    // holds copy of z at scaling point
    z: [T; 3],
}

impl<T> PowerCone<T>
where
    T: FloatT,
{
    pub fn new(α: T) -> Self {
        Self {
            α,
            H_dual: DenseMatrixSym3::zeros(),
            Hs: DenseMatrixSym3::zeros(),
            grad: [T::zero(); 3],
            z: [T::zero(); 3],
        }
    }
}

impl<T> Cone<T> for PowerCone<T>
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
        let α = self.α;

        s[0] = (T::one() + α).sqrt();
        s[1] = (T::one() + (T::one() - α)).sqrt();
        s[2] = T::zero();

        (z[0], z[1], z[2]) = (s[0], s[1], s[2]);
    }

    fn set_identity_scaling(&mut self) {
        // We should never use identity scaling because
        // we never want to allow symmetric initialization
        unimplemented!("This function should never be reached.");
    }

    fn update_scaling(&mut self, s: &[T], z: &[T], μ: T, scaling_strategy: ScalingStrategy) {
        // update both gradient and Hessian for function f*(z) at the point z
        self.update_dual_grad_H(z);

        // update the scaling matrix Hs
        self.update_Hs(s, z, μ, scaling_strategy);

        // K.z .= z
        self.z.copy_from(z);
    }

    fn Hs_is_diagonal(&self) -> bool {
        false
    }

    fn get_Hs(&self, Hsblock: &mut [T]) {
        // Hs data is already in packed triu form, so just copy
        Hsblock.copy_from(&self.Hs.data);
    }

    fn mul_Hs(&self, y: &mut [T], x: &[T], _work: &mut [T]) {
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

        // final backtracked position
        let mut wq = [T::zero(); 3];

        let _is_prim_feasible_fcn = |s: &[T]| -> bool { _is_primal_feasible_powcone(s, self.α) };
        let _is_dual_feasible_fcn = |s: &[T]| -> bool { _is_dual_feasible_powcone(s, self.α) };

        let αz = self.step_length_3d_cone(&mut wq, dz, z, αmax, αmin, step, _is_dual_feasible_fcn);
        let αs = self.step_length_3d_cone(&mut wq, ds, s, αmax, αmin, step, _is_prim_feasible_fcn);

        (αz, αs)
    }

    fn compute_barrier(&self, z: &[T], s: &[T], dz: &[T], ds: &[T], α: T) -> T {
        let mut barrier = T::zero();

        let cur_z = [z[0] + α * dz[0], z[1] + α * dz[1], z[2] + α * dz[2]];
        let cur_s = [s[0] + α * ds[0], s[1] + α * ds[1], s[2] + α * ds[2]];

        barrier += self.barrier_dual(&cur_z);
        barrier += self.barrier_primal(&cur_s);

        barrier
    }
}

// implement this marker trait to get access to
// functions common to exp and pow cones
impl<T> Nonsymmetric3DCone<T> for PowerCone<T> where T: FloatT {}

// ----------------------------------------------
//  internal operations for power cones
//
// Primal Power cone: s1^{α}s2^{1-α} ≥ s3, s1,s2 ≥ 0
// Dual Power cone: (z1/α)^{α} * (z2/(1-α))^{1-α} ≥ z3, z1,z2 ≥ 0

// Returns true if s is primal feasible
fn _is_primal_feasible_powcone<T>(s: &[T], α: T) -> bool
where
    T: FloatT,
{
    let two: T = (2f64).as_T();
    if s[0] > T::zero() && s[1] > T::zero() {
        let res =
            T::exp(two * α * s[0].logsafe() + two * (T::one() - α) * s[1].logsafe()) - s[2] * s[2];
        if res > T::zero() {
            return true;
        }
    }
    false
}

// Returns true if z is dual feasible
fn _is_dual_feasible_powcone<T>(z: &[T], α: T) -> bool
where
    T: FloatT,
{
    let two: T = (2.).as_T();

    if z[0] > T::zero() && z[1] > T::zero() {
        let res = T::exp(
            (α * two) * (z[0] / α).logsafe()
                + (T::one() - α) * (z[1] / (T::one() - α)).logsafe() * two,
        ) - z[2] * z[2];
        if res > T::zero() {
            return true;
        }
    }
    false
}

// Compute the primal gradient of f(s) at s
fn _gradient_primal<T>(s: &[T], α: T) -> [T; 3]
where
    T: FloatT,
{
    let mut g = [T::zero(); 3];
    let two: T = (2.).as_T();

    // unscaled ϕ
    let phi = (s[0]).powf(two * α) * (s[1]).powf(two - α * two);

    // obtain last element of g from the Newton-Raphson method
    let abs_s = s[2].abs();
    if abs_s > T::epsilon() {
        g[2] = _newton_raphson_powcone(abs_s, phi, α);
        if s[2] < T::zero() {
            g[2] = -g[2];
        }
        g[0] = -(α * g[2] * s[2] + T::one() + α) / s[0];
        g[1] = -((T::one() - α) * g[2] * s[2] + two - α) / s[1];
    } else {
        g[2] = T::zero();
        g[0] = -(T::one() + α) / s[0];
        g[1] = -(two - α) / s[1];
    }
    g
}

// Newton-Raphson method:
// solve a one-dimensional equation f(x) = 0
// x(k+1) = x(k) - f(x(k))/f'(x(k))
// When we initialize x0 such that 0 < x0 < x*,
// the Newton-Raphson method converges quadratically

fn _newton_raphson_powcone<T>(s3: T, phi: T, α: T) -> T
where
    T: FloatT,
{
    let two: T = (2.).as_T();
    let three: T = (3.).as_T();
    let four: T = (4.).as_T();

    // init point x0: since our dual barrier has an additional
    // shift -2α*log(α) - 2(1-α)*log(1-α) > 0 in f(x),
    // the previous selection is still feasible, i.e. f(x0) > 0
    let x0 = -s3.recip()
        + (s3 + ((phi * four) * phi / s3 / s3 + three * phi).sqrt()) * two / (phi * four - s3 * s3);

    // additional shift due to the choice of dual barrier
    let t0 = -two * α * (α.logsafe()) - two * (T::one() - α) * (T::one() - α).logsafe();

    // function for f(x) = 0
    let f0 = {
        |x: T| -> T {
            let two = (2.).as_T();
            let t1 = x * x;
            let t2 = (x * two) / s3;
            two * α * (two * α * t1 + (T::one() + α) * t2).logsafe()
                + two * (T::one() - α) * (two * (T::one() - α) * t1 + (two - α) * t2).logsafe()
                - phi.logsafe()
                - (t1 + t2).logsafe()
                - two * t2.logsafe()
                + t0
        }
    };

    // first derivative
    let f1 = {
        |x: T| -> T {
            let two = (2.).as_T();
            let t1 = x * x;
            let t2 = (two * x) / s3;
            (α * α * two) / (α * x + (T::one() + α) / s3)
                + ((T::one() - α) * two) * (T::one() - α) / ((T::one() - α) * x + (two - α) / s3)
                - ((x + s3.recip()) * two) / (t1 + t2)
        }
    };
    _newton_raphson_onesided(x0, f0, f1)
}

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

impl<T> PowerCone<T>
where
    T: FloatT,
{
    fn barrier_dual(&self, z: &[T]) -> T
    where
        T: FloatT,
    {
        // Dual barrier:
        // f*(z) = -log((z1/α)^{2α} * (z2/(1-α))^{2(1-α)} - z3*z3) - (1-α)*log(z1) - α*log(z2):
        let α = self.α;
        let two: T = (2.).as_T();
        let arg1 =
            (z[0] / α).powf(two * α) * (z[1] / (T::one() - α)).powf(two - two * α) - z[2] * z[2];

        -arg1.logsafe() - (T::one() - α) * z[0].logsafe() - α * z[1].logsafe()
    }

    fn barrier_primal(&self, s: &[T]) -> T
    where
        T: FloatT,
    {
        // Primal barrier: f(s) = ⟨s,g(s)⟩ - f*(-g(s))
        // NB: ⟨s,g(s)⟩ = -3 = - ν

        let α = self.α;
        let two: T = (2.).as_T();
        let three: T = (3.).as_T();

        let g = _gradient_primal(s, α);

        let mut out = T::zero();

        out += ((-g[0] / α).powf(two * α) * (-g[1] / (T::one() - α)).powf(two - α * two)
            - g[2] * g[2])
            .logsafe();
        out += (T::one() - α) * (-g[0]).logsafe();
        out += α * (-g[1]).logsafe() - three;
        out
    }

    fn higher_correction(&mut self, η: &mut [T; 3], ds: &[T], v: &[T])
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

        let α = self.α;
        let two: T = (2.).as_T();
        let four: T = (4.).as_T();

        let phi = (z[0] / α).powf(two * α) * (z[1] / (T::one() - α)).powf(two - two * α);
        let ψ = phi - z[2] * z[2];

        // Reuse cholH memory for further computation
        let Hψ = &mut cholH;

        η[0] = two * α * phi / z[0];
        η[1] = two * (T::one() - α) * phi / z[1];
        η[2] = -two * z[2];

        // we only need to assign the upper triangle
        // for our 3x3 symmetric type
        Hψ[(0, 1)] = four * α * (T::one() - α) * phi / (z[0] * z[1]);
        Hψ[(0, 0)] = two * α * (two * α - T::one()) * phi / (z[0] * z[0]);
        Hψ[(0, 2)] = T::zero();
        Hψ[(1, 1)] = two * (T::one() - α) * (T::one() - two * α) * phi / (z[1] * z[1]);
        Hψ[(1, 2)] = T::zero();
        Hψ[(2, 2)] = -two;

        let dotψu = u.dot(&η[..]);
        let dotψv = v.dot(&η[..]);

        let mut Hψv = [T::zero(); 3];
        Hψ.mul(&mut Hψv, v);

        let coef = (u.dot(&Hψv) * ψ - two * dotψu * dotψv) / (ψ * ψ * ψ);
        let coef2 = four
            * α
            * (two * α - T::one())
            * (T::one() - α)
            * phi
            * (u[0] / z[0] - u[1] / z[1])
            * (v[0] / z[0] - v[1] / z[1])
            / ψ;
        let inv_ψ2 = (ψ * ψ).recip();

        η[0] = coef * η[0] - two * (T::one() - α) * u[0] * v[0] / (z[0] * z[0] * z[0])
            + coef2 / z[0]
            + Hψv[0] * dotψu * inv_ψ2;

        η[1] = coef * η[1] - two * α * u[1] * v[1] / (z[1] * z[1] * z[1]) - coef2 / z[1]
            + Hψv[1] * dotψu * inv_ψ2;

        η[2] = coef * η[2] + Hψv[2] * dotψu * inv_ψ2;

        // reuse vector Hψv
        let Hψu = &mut Hψv;
        Hψ.mul(Hψu, &u);

        // @. η <= (η + Hψu*dotψv*inv_ψ2)/2
        η[..].axpby(dotψv * inv_ψ2, Hψu, T::one());
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

impl<T> PowerCone<T>
where
    T: FloatT,
{
    fn update_Hs(&mut self, s: &[T], z: &[T], μ: T, scaling_strategy: ScalingStrategy) {
        // Choose the scaling strategy
        if scaling_strategy == ScalingStrategy::Dual {
            // Dual scaling: Hs = μ*H
            self.use_dual_scaling(μ);
        } else {
            self.use_primal_dual_scaling(s, z);
        }
    }

    fn update_dual_grad_H(&mut self, z: &[T]) {
        let H = &mut self.H_dual;
        let α = self.α;
        let two: T = (2.).as_T();
        let four: T = (4.).as_T();

        let phi = (z[0] / α).powf(two * α) * (z[1] / (T::one() - α)).powf(two - two * α);
        let ψ = phi - z[2] * z[2];

        // use K.grad as a temporary workspace
        let gψ = &mut self.grad;
        gψ[0] = two * α * phi / (z[0] * ψ);
        gψ[1] = two * (T::one() - α) * phi / (z[1] * ψ);
        gψ[2] = -two * z[2] / ψ;

        // compute_Hessian(K,z,H).   Type is symmetric, so
        // only need to assign upper triangle.
        H[(0, 0)] = gψ[0] * gψ[0] - two * α * (two * α - T::one()) * phi / (z[0] * z[0] * ψ)
            + (T::one() - α) / (z[0] * z[0]);
        H[(0, 1)] = gψ[0] * gψ[1] - four * α * (T::one() - α) * phi / (z[0] * z[1] * ψ);
        H[(1, 1)] = gψ[1] * gψ[1]
            - two * (T::one() - α) * (T::one() - two * α) * phi / (z[1] * z[1] * ψ)
            + α / (z[1] * z[1]);
        H[(0, 2)] = gψ[0] * gψ[2];
        H[(1, 2)] = gψ[1] * gψ[2];
        H[(2, 2)] = gψ[2] * gψ[2] + two / ψ;

        // compute the gradient at z
        let grad = &mut self.grad;
        grad[0] = -two * α * phi / (z[0] * ψ) - (T::one() - α) / z[0];
        grad[1] = -two * (T::one() - α) * phi / (z[1] * ψ) - α / z[1];
        grad[2] = two * z[2] / ψ;
    }

    // implements dual only scaling
    fn use_dual_scaling(&mut self, μ: T) {
        self.Hs.scaled_from(μ, &self.H_dual);
    }

    //implements primal dual scaling
    fn use_primal_dual_scaling(&mut self, s: &[T], z: &[T]) {
        let three: T = (3.).as_T();

        let Hs = &mut self.Hs;
        let H_dual = &self.H_dual;

        let st = &mut self.grad;
        let mut δs = [T::zero(); 3];
        let mut tmp = [T::zero(); 3];

        // compute zt,st,μt locally
        // NB: zt,st have different sign convention wrt Mosek paper
        let zt = _gradient_primal(s, self.α);
        let dot_sz = s.dot(z);
        let μ = dot_sz / three;
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
        let de2 = H_dual.quad_form(&zt, &zt) - three * μt * μt;

        // use the primal-dual scaling
        if T::abs(de1) > T::sqrt(T::epsilon()) &&      // too close to central path
           T::abs(de2) > T::epsilon()          &&      // others for numerical stability
           dot_sz > T::zero()                  &&
           dot_δsz > T::zero()
        {
            // compute t
            // tmp = μt*st - H*zt
            H_dual.mul(&mut tmp, &zt);
            for i in 0..3 {
                tmp[i] = μt * st[i] - tmp[i]
            }

            // Hs as a workspace (only need to write the upper triangle)
            Hs.copy_from(H_dual);
            for i in 0..3 {
                for j in i..3 {
                    Hs[(i, j)] -= st[i] * st[j] / three + tmp[i] * tmp[j] / de2;
                }
            }
            let t = μ * Hs.norm_fro(); //Frobenius norm

            // generate the remaining axis
            // axis_z = cross(z,zt)
            let mut axis_z = tmp;
            axis_z[0] = z[1] * zt[2] - z[2] * zt[1];
            axis_z[1] = z[2] * zt[0] - z[0] * zt[2];
            axis_z[2] = z[0] * zt[1] - z[1] * zt[0];
            axis_z.normalize();

            // Hs = s*s'/⟨s,z⟩ + δs*δs'/⟨δs,δz⟩ + t*axis_z*axis_z'
            // (only need to write the upper triangle)
            for i in 0..3 {
                for j in i..3 {
                    Hs[(i, j)] =
                        s[i] * s[j] / dot_sz + δs[i] * δs[j] / dot_δsz + t * axis_z[i] * axis_z[j];
                }
            }

        // use the dual scaling
        } else {
            // Hs = μH when s,z are on the central path
            self.use_dual_scaling(μ);
        }
    }
}
