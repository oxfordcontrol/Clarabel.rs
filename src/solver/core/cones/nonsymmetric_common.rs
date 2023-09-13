use crate::{algebra::*, solver::core::ScalingStrategy};

// --------------------------------------
// Traits and blanket implementations for Exponential, 3D Power and ND Power Cones
// -------------------------------------
// Operations supported on all nonsymmetric cones
pub(crate) trait NonsymmetricCone<T: FloatT> {
    // Returns true if s is primal feasible
    fn is_primal_feasible(&self, s: &[T]) -> bool;

    // Returns true if z is dual feasible
    fn is_dual_feasible(&self, z: &[T]) -> bool;

    fn barrier_primal(&mut self, s: &[T]) -> T;

    fn barrier_dual(&mut self, z: &[T]) -> T;

    fn higher_correction(&mut self, η: &mut [T], ds: &[T], v: &[T]);

    fn update_dual_grad_H(&mut self, z: &[T]);
}

// --------------------------------------
// Trait and blanket utlity implementations for Exponential and 3D Power Cones
// -------------------------------------
#[allow(clippy::too_many_arguments)]

pub(crate) trait Nonsymmetric3DCone<T: FloatT> {
    fn gradient_primal(&self, s: &[T]) -> [T; 3];

    fn split_borrow_mut(
        &mut self,
    ) -> (
        &mut DenseMatrixSym3<T>,
        &mut DenseMatrixSym3<T>,
        &mut [T; 3],
        &mut [T; 3],
    );
}

pub(crate) trait Nonsymmetric3DConeUtils<T: FloatT> {
    fn update_Hs(&mut self, s: &[T], z: &[T], μ: T, scaling_strategy: ScalingStrategy);

    fn use_dual_scaling(&mut self, μ: T);

    fn use_primal_dual_scaling(&mut self, s: &[T], z: &[T]);
}

impl<T, C> Nonsymmetric3DConeUtils<T> for C
where
    T: FloatT,
    C: Nonsymmetric3DCone<T>,
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

    // implements dual only scaling
    fn use_dual_scaling(&mut self, μ: T) {
        let (H_dual, Hs, _, _) = self.split_borrow_mut();
        Hs.scaled_from(μ, H_dual);
    }

    fn use_primal_dual_scaling(&mut self, s: &[T], z: &[T]) {
        let three: T = (3.).as_T();

        let zt: [T; 3] = self.gradient_primal(s);

        let (H_dual, Hs, grad, _) = self.split_borrow_mut();

        let st = grad;
        let mut δs = [T::zero(); 3];
        let mut tmp = [T::zero(); 3];

        // compute zt,st,μt locally
        // NB: zt,st have different sign convention wrt Mosek paper
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
                tmp[i] = μt * st[i] - tmp[i];
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

// --------------------------------------
// Traits for general ND cones
// -------------------------------------

// Operations supported on ND nonsymmetrics only.  Note this
// differs from the 3D cone in particular because we don't
// return a 3D tuple for the primal gradient.
pub(crate) trait NonsymmetricNDCone<T: FloatT> {
    // Compute the primal gradient of f(s) at s
    fn gradient_primal(&self, grad: &mut [T], s: &[T]);
}

// --------------------------------------
// utility functions for nonsymmetric cones
// --------------------------------------

// find the maximum step length α≥0 so that
// q + α*dq stays in an exponential or power
// cone, or their respective dual cones.
pub(crate) fn backtrack_search<T>(
    dq: &[T],
    q: &[T],
    α_init: T,
    α_min: T,
    step: T,
    is_in_cone_fcn: impl Fn(&[T]) -> bool,
    work: &mut [T],
) -> T
where
    T: FloatT,
{
    let mut α = α_init;

    loop {
        // work = q + α*dq
        work.waxpby(T::one(), q, α, dq);

        if is_in_cone_fcn(work) {
            break;
        }
        α *= step;
        if α < α_min {
            α = T::zero();
            break;
        }
    }
    α
}
pub(crate) fn newton_raphson_onesided<T>(x0: T, f0: impl Fn(T) -> T, f1: impl Fn(T) -> T) -> T
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
