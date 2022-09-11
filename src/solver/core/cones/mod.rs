#![allow(non_snake_case)]

use crate::algebra::{FloatT, MatrixShape, VectorMath};
use enum_dispatch::*;

mod compositecone;
mod expcone;
mod nonnegativecone;
mod powcone;
mod socone;
mod zerocone;

//flatten all cone implementations to appear in this module
pub use compositecone::*;
pub use expcone::*;
pub use nonnegativecone::*;
pub use powcone::*;
pub use socone::*;
pub use zerocone::*;

use crate::solver::{core::ScalingStrategy, CoreSettings};

#[enum_dispatch]
pub trait Cone<T>
where
    T: FloatT,
{
    // functions relating to basic sizing
    fn dim(&self) -> usize;
    fn degree(&self) -> usize;
    fn numel(&self) -> usize;

    fn is_symmetric(&self) -> bool;

    // converts an elementwise scaling into
    // a scaling that preserves cone memership
    fn rectify_equilibration(&self, δ: &mut [T], e: &[T]) -> bool;

    // functions relating to unit vectors and cone initialization
    fn shift_to_cone(&self, z: &mut [T]);
    fn unit_initialization(&self, z: &mut [T], s: &mut [T]);

    // Compute scaling points
    fn set_identity_scaling(&mut self);
    fn update_scaling(&mut self, s: &[T], z: &[T], μ: T, scaling_strategy: ScalingStrategy);

    // operations on the Hessian of the centrality condition
    // : W^TW for symmmetric cones
    // : μH(s) for nonsymmetric cones
    fn Hs_is_diagonal(&self) -> bool;
    fn get_Hs(&self, Hsblock: &mut [T]);
    fn mul_Hs(&self, y: &mut [T], x: &[T], work: &mut [T]);

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
    // The "offset" in Δs_from_Δz_offset! is then just ds
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
    // The "offset" in Δs_from_Δz_offset! is then Wᵀ(λ \ ds)
    //
    // Note that the Δs_from_Δz_offset! function is only needed in the
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
    fn Δs_from_Δz_offset(&self, out: &mut [T], ds: &[T], work: &mut [T]);

    // Find the maximum step length in some search direction
    fn step_length(
        &self,
        dz: &[T],
        ds: &[T],
        z: &[T],
        s: &[T],
        settings: &CoreSettings<T>,
        αmax: T,
    ) -> (T, T);

    // return the barrier function at (z+αdz,s+αds)
    fn compute_barrier(&self, z: &[T], s: &[T], dz: &[T], ds: &[T], α: T) -> T;
}

// Operations supported on symmetric cones only
pub trait SymmetricCone<T: FloatT>: JordanAlgebra<T> {
    // Add the scaled identity element e
    fn add_scaled_e(&self, x: &mut [T], α: T);

    // Multiplication by the scaling point
    fn mul_W(&self, is_transpose: MatrixShape, y: &mut [T], x: &[T], α: T, β: T);
    fn mul_Winv(&self, is_transpose: MatrixShape, y: &mut [T], x: &[T], α: T, β: T);

    // x = λ \ z
    // Included as a special case since q \ z for general q is difficult
    // to implement for general q i PSD cone and never actually needed.
    fn λ_inv_circ_op(&self, x: &mut [T], z: &[T]);
}

pub trait JordanAlgebra<T: FloatT> {
    fn circ_op(&self, x: &mut [T], y: &[T], z: &[T]);
    fn inv_circ_op(&self, x: &mut [T], y: &[T], z: &[T]);
}

// Marker trait for 3 dimensional nonsymmetric cones, i.e. exp and pow cones
pub trait Nonsymmetric3DCone<T: FloatT> {}

// --------------------------------------
// Trait with blanket implementation for all symmetric cones
// Provides functions that are identical across types

pub(super) trait SymmetricConeUtils<T: FloatT> {
    fn _combined_ds_shift_symmetric(
        &self,
        shift: &mut [T],
        step_z: &mut [T],
        step_s: &mut [T],
        σμ: T,
    );
    fn _Δs_from_Δz_offset_symmetric(&self, out: &mut [T], ds: &[T], work: &mut [T]);
}

impl<T, C> SymmetricConeUtils<T> for C
where
    T: FloatT,
    C: SymmetricCone<T>,
{
    // compute shift in the combined step :
    //     λ ∘ (WΔz + W^{-⊤}Δs) = - (affine_ds + shift)
    // The affine term (computed in affine_ds!) is λ ∘ λ
    // The shift term is W⁻¹Δs ∘ WΔz - σμe

    fn _combined_ds_shift_symmetric(
        &self,
        shift: &mut [T],
        step_z: &mut [T],
        step_s: &mut [T],
        σμ: T,
    ) {
        // The shift must be assembled carefully if we want to be economical with
        // allocated memory.  Will modify the step.z and step.s in place since
        // they are from the affine step and not needed anymore.
        //
        // We can't have aliasing vector arguments to gemv_W or gemv_Winv, so
        // we need a temporary variable to assign #Δz <= WΔz and Δs <= W⁻¹Δs

        // shift vector used as workspace for a few steps
        let tmp = shift;

        //PJG : order of arguments is now like Julia, but it fails because
        //step_z and step_s must be taken as mutable so that I can modify them
        //in place.

        //Δz <- Wdz
        tmp.copy_from(step_z);
        self.mul_W(MatrixShape::N, step_z, tmp, T::one(), T::zero());

        //Δs <- W⁻¹Δs
        tmp.copy_from(step_s);
        self.mul_Winv(MatrixShape::T, step_s, tmp, T::one(), T::zero());

        //shift = W⁻¹Δs ∘ WΔz - σμe
        let shift = tmp;
        self.circ_op(shift, step_s, step_z);
        self.add_scaled_e(shift, -σμ);
    }

    // compute the constant part of Δs when written as a function of Δz
    // in the solution of a KKT system

    fn _Δs_from_Δz_offset_symmetric(&self, out: &mut [T], ds: &[T], work: &mut [T]) {
        //tmp = λ \ ds
        self.λ_inv_circ_op(work, ds);

        //out = Wᵀ(λ \ ds) = Wᵀ(work)
        self.mul_W(MatrixShape::T, out, work, T::one(), T::zero());
    }
}

// --------------------------------------
// Trait with blanket implementation for the 3 dimensional
// Exp and Pow cones
#[allow(clippy::too_many_arguments)]
pub(super) trait Nonsymmetric3DConeUtils<T: FloatT> {
    fn step_length_3d_cone(
        &self,
        wq: &mut [T],
        dq: &[T],
        q: &[T],
        α_init: T,
        α_min: T,
        backtrack: T,
        is_in_cone_fcn: impl Fn(&[T]) -> bool,
    ) -> T;
}

impl<T, C> Nonsymmetric3DConeUtils<T> for C
where
    T: FloatT,
    C: Nonsymmetric3DCone<T>,
{
    // find the maximum step length α≥0 so that
    // q + α*dq stays in an exponential or power
    // cone, or their respective dual cones.
    //
    // NB: Not for use as a general checking
    // function because cone lengths are hardcoded
    // to R^3 for faster execution.

    fn step_length_3d_cone(
        &self,
        wq: &mut [T],
        dq: &[T],
        q: &[T],
        α_init: T,
        α_min: T,
        backtrack: T,
        is_in_cone_fcn: impl Fn(&[T]) -> bool,
    ) -> T {
        let mut α = α_init;

        loop {
            // wq = q + α*dq
            for i in 0..3 {
                wq[i] = q[i] + α * dq[i];
            }

            if is_in_cone_fcn(wq) {
                break;
            }
            α *= backtrack;
            if α < α_min {
                α = T::zero();
                break;
            }
        }
        α
    }
}

// ---------------------------------------------------
// We define some machinery here for enumerating the
// different cone types that can live in the composite cone
// symbol
// ---------------------------------------------------

use core::hash::{Hash, Hasher};
use std::{cmp::PartialEq, mem::discriminant};

/// API type describing the type of a conic constraint.
///  
#[derive(Debug, Clone, Copy)]
pub enum SupportedConeT<T> {
    /// The zero cone (used for equality constraints).
    ///
    /// The parameter indicates the cones dimension.
    ZeroConeT(usize),
    /// The nonnegative orthant.  
    ///
    /// The parameter indicates the cones dimension.
    NonnegativeConeT(usize),
    /// The second order cone / Lorenz cone / ice-cream cone.
    ///  
    /// The parameter indicates the cones dimension.
    SecondOrderConeT(usize),
    /// The exponential cone in R^3.
    ///
    /// This cone takes no parameters
    ExponentialConeT(),
    /// The power cone in R^3.
    ///
    /// The parameter indicates the power.
    PowerConeT(T),
}

impl<T> SupportedConeT<T> {
    /// Returns the name of the cone from its enum.  Used for printing progress.

    pub fn variant_name(&self) -> &'static str {
        match self {
            SupportedConeT::ZeroConeT(_) => "ZeroConeT",
            SupportedConeT::NonnegativeConeT(_) => "NonnegativeConeT",
            SupportedConeT::SecondOrderConeT(_) => "SecondOrderConeT",
            SupportedConeT::ExponentialConeT() => "ExponentialConeT",
            SupportedConeT::PowerConeT(_) => "PowerConeT",
        }
    }

    // this reports the number of slack variables that
    //s will be generated by this cone.  Equivalent to
    // `numels` for the internal cone representation

    pub(crate) fn nvars(&self) -> usize {
        match self {
            SupportedConeT::ZeroConeT(dim) => *dim,
            SupportedConeT::NonnegativeConeT(dim) => *dim,
            SupportedConeT::SecondOrderConeT(dim) => *dim,
            SupportedConeT::ExponentialConeT() => 3,
            SupportedConeT::PowerConeT(_) => 3,
            // For PSDTriangleT, we will need
            // (dim*(dim+1)) >> 1
        }
    }
}

impl<T> std::fmt::Display for SupportedConeT<T>
where
    T: FloatT,
{
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", &self.variant_name().to_string())
    }
}

// we will use the SupportedConeT as a user facing marker
// for the constraint types, and then map them through
// a dictionary to get the internal cone representations.
// we will also make a HashMap of cone type counts, so need
// to define custom hashing and comparator ops
impl<T> Eq for SupportedConeT<T> {}
impl<T> PartialEq for SupportedConeT<T> {
    fn eq(&self, other: &Self) -> bool {
        discriminant(self) == discriminant(other)
    }
}

impl<T> Hash for SupportedConeT<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        discriminant(self).hash(state);
    }
}

// -------------------------------------
// Here we make a corresponding internal ConeEnum type that
// uses enum_dispatch to allow for static dispatching against
// all of our internal cone types
// -------------------------------------

#[allow(clippy::enum_variant_names)]
#[enum_dispatch(Cone<T>)]
pub enum SupportedCone<T>
where
    T: FloatT,
{
    ZeroCone(ZeroCone<T>),
    NonnegativeCone(NonnegativeCone<T>),
    SecondOrderCone(SecondOrderCone<T>),
    ExponentialCone(ExponentialCone<T>),
    PowerCone(PowerCone<T>),
}

pub fn make_cone<T: FloatT>(cone: SupportedConeT<T>) -> SupportedCone<T> {
    match cone {
        SupportedConeT::NonnegativeConeT(dim) => NonnegativeCone::<T>::new(dim).into(),
        SupportedConeT::ZeroConeT(dim) => ZeroCone::<T>::new(dim).into(),
        SupportedConeT::SecondOrderConeT(dim) => SecondOrderCone::<T>::new(dim).into(),
        SupportedConeT::ExponentialConeT() => ExponentialCone::<T>::new().into(),
        SupportedConeT::PowerConeT(α) => PowerCone::<T>::new(α).into(),
    }
}
