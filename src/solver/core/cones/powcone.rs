use super::Cone;
use crate::{algebra::*, solver::CoreSettings};

// -------------------------------------
// Power Cone
// -------------------------------------

pub struct PowerCone<T: FloatT = f64> {
    //power
    pow: T,
    // PLACEHOLDERS
    foo: Vec<T>,
}

impl<T> PowerCone<T>
where
    T: FloatT,
{
    pub fn new(pow: T) -> Self {
        Self {
            foo: vec![T::zero(); 3],
            pow,
        }
    }
}

impl<T> Cone<T> for PowerCone<T>
where
    T: FloatT,
{
    fn dim(&self) -> usize {
        todo!()
    }

    fn degree(&self) -> usize {
        todo!()
    }

    fn numel(&self) -> usize {
        todo!()
    }

    fn is_symmetric(&self) -> usize {
        false
    }

    fn rectify_equilibration(&self, δ: &mut [T], e: &[T]) -> bool {
        todo!()
    }

    fn shift_to_cone(&self, z: &mut [T]) {
        todo!()
    }

    fn unit_initialization(&self, z: &mut [T], s: &mut [T]) {
        todo!()
    }

    fn set_identity_scaling(&mut self) {
        todo!()
    }

    fn update_scaling(
        &mut self,
        s: &[T],
        z: &[T],
        μ: T,
        scaling_strategy: crate::solver::core::ScalingStrategy,
    ) {
        todo!()
    }

    fn Hs_is_diagonal(&self) -> bool {
        todo!()
    }

    fn get_Hs(&self, Hsblock: &mut [T]) {
        todo!()
    }

    fn mul_Hs(&self, y: &mut [T], x: &[T], work: &mut [T]) {
        todo!()
    }

    fn affine_ds(&self, ds: &mut [T], s: &[T]) {
        todo!()
    }

    fn combined_ds_shift(&self, shift: &mut [T], step_z: &[T], step_s: &[T], σμ: T) {
        todo!()
    }

    fn Δs_from_Δz_offset(&self, out: &mut [T], ds: &[T], work: &mut [T]) {
        todo!()
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
        todo!()
    }

    fn compute_barrier(&self, z: &[T], s: &[T], dz: &[T], ds: &[T], α: T) -> T {
        todo!()
    }
}
