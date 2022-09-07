use super::Cone;
use crate::{
    algebra::*,
    solver::{core::ScalingStrategy, CoreSettings},
};
use core::marker::PhantomData;

// -------------------------------------
// Zero Cone
// -------------------------------------

pub struct ZeroCone<T: FloatT = f64> {
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
    fn dim(&self) -> usize {
        self.dim
    }

    fn degree(&self) -> usize {
        0
    }

    fn numel(&self) -> usize {
        self.dim()
    }

    fn is_symmetric(&self) -> bool {
        true
    }

    fn rectify_equilibration(&self, δ: &mut [T], e: &[T]) -> bool {
        δ.copy_from(e);
        false
    }

    fn shift_to_cone(&self, z: &mut [T]) {
        z.fill(T::zero());
    }

    fn unit_initialization(&self, z: &mut [T], s: &mut [T]) {
        s.fill(T::zero());
        z.fill(T::zero());
    }

    fn set_identity_scaling(&mut self) {
        //nothing to do
    }

    fn update_scaling(&mut self, _s: &[T], _z: &[T], _μ: T, _scaling_strategy: ScalingStrategy) {
        //nothing to do
    }

    fn Hs_is_diagonal(&self) -> bool {
        true
    }

    fn get_Hs(&self, Hsblock: &mut [T]) {
        Hsblock.fill(T::zero());
    }

    fn mul_Hs(&self, y: &mut [T], _x: &[T], _work: &mut [T]) {
        y.fill(T::zero());
    }

    fn affine_ds(&self, ds: &mut [T], _s: &[T]) {
        ds.fill(T::zero());
    }

    fn combined_ds_shift(&mut self, shift: &mut [T], _step_z: &[T], _step_s: &[T], _σμ: T) {
        shift.fill(T::zero());
    }

    fn Δs_from_Δz_offset(&self, out: &mut [T], _ds: &[T], _work: &mut [T]) {
        out.fill(T::zero());
    }

    fn step_length(
        &self,
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

    fn compute_barrier(&self, _z: &[T], _s: &[T], _dz: &[T], _ds: &[T], _α: T) -> T {
        T::zero()
    }
}
