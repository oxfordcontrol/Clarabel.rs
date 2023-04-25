use crate::algebra::{FloatT, MatrixShape, VectorMath};

use super::*;

// --------------------------------------
// Traits and blanket implementations for Exponential and PowerCones
// -------------------------------------

// Operations supported on symmetric cones only
pub trait SymmetricCone<T: FloatT>: JordanAlgebra<T> {
    // Multiplication by the scaling point
    fn mul_W(&mut self, is_transpose: MatrixShape, y: &mut [T], x: &[T], α: T, β: T);
    fn mul_Winv(&mut self, is_transpose: MatrixShape, y: &mut [T], x: &[T], α: T, β: T);

    // x = λ \ z
    // Included as a special case since q \ z for general q is difficult
    // to implement for general q i PSD cone and never actually needed.
    fn λ_inv_circ_op(&mut self, x: &mut [T], z: &[T]);
}

pub trait JordanAlgebra<T: FloatT> {
    fn circ_op(&mut self, x: &mut [T], y: &[T], z: &[T]);
    fn inv_circ_op(&mut self, x: &mut [T], y: &[T], z: &[T]);
}

// --------------------------------------
// Trait with blanket implementation for all symmetric cones
// Provides functions that are identical across types

pub(super) trait SymmetricConeUtils<T: FloatT> {
    fn _combined_ds_shift_symmetric(
        &mut self,
        shift: &mut [T],
        step_z: &mut [T],
        step_s: &mut [T],
        σμ: T,
    );
    fn _Δs_from_Δz_offset_symmetric(&mut self, out: &mut [T], ds: &[T], work: &mut [T]);
}

impl<T, C> SymmetricConeUtils<T> for C
where
    T: FloatT,
    C: SymmetricCone<T> + Cone<T>,
{
    // compute shift in the combined step :
    //     λ ∘ (WΔz + W^{-⊤}Δs) = - (affine_ds + shift)
    // The affine term (computed in affine_ds!) is λ ∘ λ
    // The shift term is W⁻¹Δs ∘ WΔz - σμe

    fn _combined_ds_shift_symmetric(
        &mut self,
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

        //Δz <- WΔz
        tmp.copy_from(step_z);
        self.mul_W(MatrixShape::N, step_z, tmp, T::one(), T::zero());

        //Δs <- W⁻¹Δs
        tmp.copy_from(step_s);
        self.mul_Winv(MatrixShape::T, step_s, tmp, T::one(), T::zero());

        //shift = W⁻¹Δs ∘ WΔz - σμe
        let shift = tmp;
        self.circ_op(shift, step_s, step_z);

        //cone will be self dual, so Primal/Dual not important
        self.scaled_unit_shift(shift, -σμ, PrimalOrDualCone::PrimalCone);
    }

    // compute the constant part of Δs when written as a function of Δz
    // in the solution of a KKT system

    fn _Δs_from_Δz_offset_symmetric(&mut self, out: &mut [T], ds: &[T], work: &mut [T]) {
        //tmp = λ \ ds
        self.λ_inv_circ_op(work, ds);

        //out = Wᵀ(λ \ ds) = Wᵀ(work)
        self.mul_W(MatrixShape::T, out, work, T::one(), T::zero());
    }
}
