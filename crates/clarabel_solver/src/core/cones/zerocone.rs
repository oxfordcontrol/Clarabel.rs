use super::Cone;
use clarabel_algebra::*;
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

    fn rectify_equilibration(&self, δ: &mut [T], e: &[T]) -> bool {
        δ.copy_from(e);
        false
    }

    fn WtW_is_diagonal(&self) -> bool {
        true
    }

    fn update_scaling(&mut self, _s: &[T], _z: &[T]) {
        //nothing to do
    }

    fn set_identity_scaling(&mut self) {
        //nothing to do
    }

    fn λ_circ_λ(&self, x: &mut [T]) {
        x.fill(T::zero());
    }

    fn circ_op(&self, x: &mut [T], _y: &[T], _z: &[T]) {
        x.fill(T::zero());
    }

    fn λ_inv_circ_op(&self, x: &mut [T], _z: &[T]) {
        x.fill(T::zero());
    }

    fn inv_circ_op(&self, x: &mut [T], _y: &[T], _z: &[T]) {
        x.fill(T::zero());
    }

    fn shift_to_cone(&self, z: &mut [T]) {
        z.fill(T::zero());
    }

    fn get_WtW_block(&self, WtWblock: &mut [T]) {
        WtWblock.fill(T::zero());
    }

    fn gemv_W(&self, _is_transpose: MatrixShape, _x: &[T], y: &mut [T], _α: T, β: T) {
        //treat W like zero
        y.scale(β);
    }

    fn gemv_Winv(&self, _is_transpose: MatrixShape, _x: &[T], y: &mut [T], _α: T, β: T) {
        //treat Winv like zero
        y.scale(β);
    }

    fn add_scaled_e(&self, _x: &mut [T], _α: T) {
        //e = 0, do nothing
    }

    fn step_length(&self, _dz: &[T], _ds: &[T], _z: &[T], _s: &[T]) -> (T, T) {
        //equality constraints allow arbitrary step length
        let huge = T::max_value();
        (huge, huge)
    }
}
