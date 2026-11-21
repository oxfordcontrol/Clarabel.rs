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

        let st: &[T; 3] = grad;
        let mut δs = [T::zero(), T::zero(), T::zero()];
        let mut tmp = [T::zero(), T::zero(), T::zero()];

        // compute zt,st,μt locally
        // NB: zt,st have different sign convention wrt Mosek paper
        let dot_sz = s.dot(z);
        let μ = dot_sz.clone() / three.clone();
        let μt = st[..].dot(&zt[..]) / three.clone();

        // δs = s + μ*st
        // δz = z + μ*zt
        let mut δz = tmp.clone();
        for i in 0..3 {
            δs[i] = s[i].clone() + μ.clone() * st[i].clone();
            δz[i] = z[i].clone() + μ.clone() * zt[i].clone();
        }
        let dot_δsz = δs[..].dot(&δz[..]);

        let de1 = μ.clone() * μt.clone() - T::one();
        let de2 = H_dual.quad_form(&zt, &zt) - three.clone() * μt.clone() * μt.clone();

        // use the primal-dual scaling
        if de1.abs() > T::sqrt(T::epsilon()) &&      // too close to central path
           de2.abs() > T::epsilon()          &&      // others for numerical stability
           dot_sz > T::zero()                  &&
           dot_δsz > T::zero()
        {
            // compute t
            // tmp = μt*st - H*zt
            H_dual.mul(&mut tmp, &zt);
            for i in 0..3 {
                tmp[i] = μt.clone() * st[i].clone() - tmp[i].clone();
            }

            // Hs as a workspace (only need to write the upper triangle)
            Hs.copy_from(H_dual);
            for i in 0..3 {
                for j in i..3 {
                    let cur: T = Hs[(i, j)].clone();
                    Hs[(i, j)] = cur
                        - (st[i].clone() * st[j].clone() / three.clone()
                            + tmp[i].clone() * tmp[j].clone() / de2.clone());
                }
            }
            let t = μ.clone() * Hs.norm_fro(); //Frobenius norm

            // generate the remaining axis
            // axis_z = cross(z,zt)
            let mut axis_z = tmp.clone();
            axis_z[0] = z[1].clone() * zt[2].clone() - z[2].clone() * zt[1].clone();
            axis_z[1] = z[2].clone() * zt[0].clone() - z[0].clone() * zt[2].clone();
            axis_z[2] = z[0].clone() * zt[1].clone() - z[1].clone() * zt[0].clone();
            axis_z.normalize();

            // Hs = s*s'/⟨s,z⟩ + δs*δs'/⟨δs,δz⟩ + t*axis_z*axis_z'
            // (only need to write the upper triangle)
            for i in 0..3 {
                for j in i..3 {
                    Hs[(i, j)] = s[i].clone() * s[j].clone() / dot_sz.clone()
                        + δs[i].clone() * δs[j].clone() / dot_δsz.clone()
                        + t.clone() * axis_z[i].clone() * axis_z[j].clone();
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
        work.waxpby(T::one(), q, α.clone(), dq);

        if is_in_cone_fcn(work) {
            break;
        }
        α *= step.clone();
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
        let dfdx = f1(x.clone());
        let dx = -f0(x.clone()) / dfdx.clone();

        if (dx < T::epsilon())
            || ((dx.clone() / x.clone()).abs() < T::sqrt(T::epsilon()))
            || (dfdx.abs() < T::epsilon())
        {
            break;
        }
        x += dx;
    }

    x
}
