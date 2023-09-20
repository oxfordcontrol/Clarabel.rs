#![allow(non_snake_case)]

use crate::algebra::FloatT;
use crate::solver::{core::ScalingStrategy, CoreSettings};
use enum_dispatch::*;

// the supported cone wrapper type for primitives
// and the composite cone
mod compositecone;
mod supportedcone;
// primitive cone types
mod expcone;
mod genpowcone;
mod nonnegativecone;
mod powcone;
mod socone;
mod zerocone;
// partially specialized traits and blanket implementataions
mod nonsymmetric_common;
mod symmetric_common;

//re-export everything to appear as one module
use nonsymmetric_common::*;
pub use {
    compositecone::*, expcone::*, genpowcone::*, nonnegativecone::*, powcone::*, socone::*,
    supportedcone::*, symmetric_common::*, zerocone::*,
};

// only use PSD cones with SDP/Blas enabled
#[cfg(feature = "sdp")]
mod psdtrianglecone;
#[cfg(feature = "sdp")]
pub use psdtrianglecone::*;

// marker for primal / dual distinctions
#[derive(Eq, PartialEq, Clone, Debug, Copy)]
pub enum PrimalOrDualCone {
    PrimalCone,
    DualCone,
}

#[enum_dispatch]
pub trait Cone<T>
where
    T: FloatT,
{
    // functions relating to basic sizing
    fn degree(&self) -> usize;
    fn numel(&self) -> usize;

    //Can the cone provide a sparse expanded representation?
    fn is_sparse_expandable(&self) -> bool;

    // is the cone symmetric?  NB: zero cone still reports true
    fn is_symmetric(&self) -> bool;

    // report false here if only dual scaling is implemented (e.g. GenPowerCone)
    fn allows_primal_dual_scaling(&self) -> bool;

    // converts an elementwise scaling into
    // a scaling that preserves cone memership
    fn rectify_equilibration(&self, δ: &mut [T], e: &[T]) -> bool;

    // returns (α,β) such that:
    // z - α⋅e is just on the cone boundary, with value
    // α >=0 indicates z \in cone, i.e. negative margin ===
    // outside of the cone.
    //
    // β is the sum of the margins that are positive.   For most
    // cones this will just be β = max(0.,α), but for cones that
    // are composites (e.g. the R_n^+), it is the sum of all of
    // the positive margin terms.
    fn margins(&mut self, z: &mut [T], pd: PrimalOrDualCone) -> (T, T);

    // functions relating to unit vectors and cone initialization
    fn scaled_unit_shift(&self, z: &mut [T], α: T, pd: PrimalOrDualCone);
    fn unit_initialization(&self, z: &mut [T], s: &mut [T]);

    // Compute scaling points
    fn set_identity_scaling(&mut self);
    fn update_scaling(
        &mut self, s: &[T], z: &[T], μ: T, scaling_strategy: ScalingStrategy
    ) -> bool;

    // operations on the Hessian of the centrality condition
    // : W^TW for symmmetric cones
    // : μH(s) for nonsymmetric cones
    fn Hs_is_diagonal(&self) -> bool;
    fn get_Hs(&self, Hsblock: &mut [T]);
    fn mul_Hs(&mut self, y: &mut [T], x: &[T], work: &mut [T]);

    // ---------------------------------------------------------
    // Linearized centrality condition functions
    //
    // For nonsymmetric cones:
    // -----------------------
    //
    // The centrality condition is : s = -μg(z)
    //
    // The linearized version is :
    //     Δs + μH(z)Δz = -ds = -(affine_ds + combined_ds_shift)
    //
    // The affine term (computed in affine_ds!) is s
    // The shift term is μg(z) plus any higher order corrections
    //
    // # To recover Δs from Δz, we can write
    //     Δs = - (ds + μHΔz)
    // The "offset" in Δs_from_Δz_offset is then just ds
    //
    // For symmetric cones:
    // --------------------
    //
    // The centrality condition is : (W(z + Δz) ∘ W⁻ᵀ(s + Δs) = μe
    //
    // The linearized version is :
    //     λ ∘ (WΔz + WᵀΔs) = -ds = - (affine_ds + combined_ds_shift)
    //
    // The affine term (computed in affine_ds!) is λ ∘ λ
    // The shift term is W⁻¹Δs_aff ∘ WΔz_aff - σμe, where the terms
    // Δs_aff an Δz_aff are from the affine KKT solve, i.e. they
    // are the Mehrotra correction terms.
    //
    // To recover Δs from Δz, we can write
    //     Δs = - ( Wᵀ(λ \ ds) + WᵀW Δz)
    // The "offset" in Δs_from_Δz_offset is then Wᵀ(λ \ ds)
    //
    // Note that the Δs_from_Δz_offset function is only needed in the
    // general combined step direction.   In the affine step direction,
    // we have the identity Wᵀ(λ \ (λ ∘ λ )) = s.  The symmetric and
    // nonsymmetric cases coincide and offset is taken directly as s.
    //
    // The affine step directions terms steps_z and step_s are
    // passed to combined_ds_shift as mutable.  Once they have been
    // used to compute the combined ds shift they are no longer needed,
    // so may be modified in place as workspace.
    // ---------------------------------------------------------
    fn affine_ds(&self, ds: &mut [T], s: &[T]);
    fn combined_ds_shift(&mut self, shift: &mut [T], step_z: &mut [T], step_s: &mut [T], σμ: T);
    fn Δs_from_Δz_offset(&mut self, out: &mut [T], ds: &[T], work: &mut [T], z: &[T]);

    // Find the maximum step length in some search direction
    fn step_length(
        &mut self,
        dz: &[T],
        ds: &[T],
        z: &[T],
        s: &[T],
        settings: &CoreSettings<T>,
        αmax: T,
    ) -> (T, T);

    // return the barrier function at (z+αdz,s+αds)
    fn compute_barrier(&mut self, z: &[T], s: &[T], dz: &[T], ds: &[T], α: T) -> T;
}
