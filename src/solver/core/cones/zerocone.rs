use super::*;
use crate::{
    algebra::*,
    solver::{core::ScalingStrategy, CoreSettings},
};
use core::marker::PhantomData;

// -------------------------------------
// Zero Cone
// -------------------------------------

pub struct ZeroCone<T> {
    dim: usize,
    phantom: PhantomData<T>,
}

impl<T> ZeroCone<T>
where
    T: FloatT,
{
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            phantom: PhantomData,
        }
    }
}

impl<T> Cone<T> for ZeroCone<T>
where
    T: FloatT,
{
    fn degree(&self) -> usize {
        0
    }

    fn numel(&self) -> usize {
        self.dim
    }

    fn is_symmetric(&self) -> bool {
        true
    }

    fn is_sparse_expandable(&self) -> bool {
        false
    }

    fn allows_primal_dual_scaling(&self) -> bool {
        true
    }

    fn rectify_equilibration(&self, δ: &mut [T], _e: &[T]) -> bool {
        δ.set(T::one());
        false
    }

    fn margins(&mut self, _z: &mut [T], _pd: PrimalOrDualCone) -> (T, T) {
        // for either primal or dual case we specify infinite
        // minimum margin and zero total margin.
        // if we later shift a vector into the zero cone
        // using scaled_unit_shift!, we just zero it
        // out regardless of the applied shift anway
        (T::max_value(), T::zero())
    }
    fn scaled_unit_shift(&self, z: &mut [T], _α: T, pd: PrimalOrDualCone) {
        if pd == PrimalOrDualCone::PrimalCone {
            z.fill(T::zero());
        } else {
            // do nothing
        }
    }

    fn unit_initialization(&self, z: &mut [T], s: &mut [T]) {
        s.fill(T::zero());
        z.fill(T::zero());
    }

    fn set_identity_scaling(&mut self) {
        //nothing to do
    }

    fn update_scaling(
        &mut self,
        _s: &[T],
        _z: &[T],
        _μ: T,
        _scaling_strategy: ScalingStrategy,
    ) -> bool {
        true
    }

    fn Hs_is_diagonal(&self) -> bool {
        true
    }

    fn get_Hs(&self, Hsblock: &mut [T]) {
        Hsblock.fill(T::zero());
    }

    fn mul_Hs(&mut self, y: &mut [T], _x: &[T], _work: &mut [T]) {
        y.fill(T::zero());
    }

    fn affine_ds(&self, ds: &mut [T], _s: &[T]) {
        ds.fill(T::zero());
    }

    fn combined_ds_shift(
        &mut self, shift: &mut [T], _step_z: &mut [T], _step_s: &mut [T], _σμ: T
    ) {
        shift.fill(T::zero());
    }

    fn Δs_from_Δz_offset(&mut self, out: &mut [T], _ds: &[T], _work: &mut [T], _z: &[T]) {
        out.fill(T::zero());
    }

    fn step_length(
        &mut self,
        _dz: &[T],
        _ds: &[T],
        _z: &[T],
        _s: &[T],
        _settings: &CoreSettings<T>,
        αmax: T,
    ) -> (T, T) {
        //equality constraints allow arbitrary step length
        (αmax, αmax)
    }

    fn compute_barrier(&mut self, _z: &[T], _s: &[T], _dz: &[T], _ds: &[T], _α: T) -> T {
        T::zero()
    }
}
