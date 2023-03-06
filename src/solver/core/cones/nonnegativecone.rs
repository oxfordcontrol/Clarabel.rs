use super::*;
use crate::{
    algebra::*,
    solver::{core::ScalingStrategy, CoreSettings},
};

// -------------------------------------
// Nonnegative Cone
// -------------------------------------

pub struct NonnegativeCone<T: FloatT = f64> {
    dim: usize,
    w: Vec<T>,
    λ: Vec<T>,
}

impl<T> NonnegativeCone<T>
where
    T: FloatT,
{
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            w: vec![T::zero(); dim],
            λ: vec![T::zero(); dim],
        }
    }
}

impl<T> Cone<T> for NonnegativeCone<T>
where
    T: FloatT,
{
    fn dim(&self) -> usize {
        self.dim
    }

    fn degree(&self) -> usize {
        self.dim()
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

    fn margins(&self, z: &mut [T], _pd: PrimalOrDualCone) -> (T, T) {
        let α = z.minimum();
        let β = z.iter().fold(T::zero(), |β, &zi| β + T::max(zi, T::zero()));
        (α, β)
    }

    fn scaled_unit_shift(&self, z: &mut [T], α: T, _pd: PrimalOrDualCone) {
        z.translate(α);
    }

    fn unit_initialization(&self, z: &mut [T], s: &mut [T]) {
        z.fill(T::one());
        s.fill(T::one());
    }

    fn set_identity_scaling(&mut self) {
        self.w.fill(T::one());
    }

    fn update_scaling(
        &mut self,
        s: &[T],
        z: &[T],
        _μ: T,
        _scaling_strategy: ScalingStrategy,
    ) -> bool {
        let λw = self.λ.iter_mut().zip(self.w.iter_mut());
        let sz = s.iter().zip(z.iter());

        for ((λ, w), (s, z)) in λw.zip(sz) {
            *λ = T::sqrt((*s) * (*z));
            *w = T::sqrt((*s) / (*z));
        }

        true
    }

    fn Hs_is_diagonal(&self) -> bool {
        true
    }

    fn get_Hs(&self, Hsblock: &mut [T]) {
        assert_eq!(self.w.len(), Hsblock.len());
        for (blki, &wi) in Hsblock.iter_mut().zip(self.w.iter()) {
            *blki = wi * wi;
        }
    }

    fn mul_Hs(&self, y: &mut [T], x: &[T], _work: &mut [T]) {
        //NB : seemingly sensitive to order of multiplication
        for (yi, (&wi, &xi)) in y.iter_mut().zip(self.w.iter().zip(x)) {
            *yi = wi * (wi * xi);
        }
    }

    fn affine_ds(&self, ds: &mut [T], _s: &[T]) {
        assert_eq!(self.λ.len(), ds.len());
        for (dsi, &λi) in ds.iter_mut().zip(self.λ.iter()) {
            *dsi = λi * λi;
        }
    }

    fn combined_ds_shift(&mut self, dz: &mut [T], step_z: &mut [T], step_s: &mut [T], σμ: T) {
        //PJG: could be done faster for nonnegatives?
        self._combined_ds_shift_symmetric(dz, step_z, step_s, σμ);
    }

    fn Δs_from_Δz_offset(&self, out: &mut [T], ds: &[T], _work: &mut [T], z: &[T]) {
        for (outi, (&dsi, &zi)) in out.iter_mut().zip(ds.iter().zip(z)) {
            *outi = dsi / zi;
        }
    }

    fn step_length(
        &self,
        dz: &[T],
        ds: &[T],
        z: &[T],
        s: &[T],
        _settings: &CoreSettings<T>,
        αmax: T,
    ) -> (T, T) {
        assert_eq!(z.len(), s.len());
        assert_eq!(dz.len(), z.len());
        assert_eq!(ds.len(), s.len());

        let mut αz = αmax;
        let mut αs = αmax;

        for i in 0..z.len() {
            if dz[i] < T::zero() {
                αz = T::min(αz, -z[i] / dz[i]);
            }
            if ds[i] < T::zero() {
                αs = T::min(αs, -s[i] / ds[i]);
            }
        }
        (αz, αs)
    }

    fn compute_barrier(&self, z: &[T], s: &[T], dz: &[T], ds: &[T], α: T) -> T {
        assert_eq!(z.len(), s.len());
        assert_eq!(dz.len(), z.len());
        assert_eq!(ds.len(), s.len());
        let mut barrier = T::zero();
        let s_ds = s.iter().zip(ds.iter());
        let z_dz = z.iter().zip(dz.iter());
        for ((&s, &ds), (&z, &dz)) in s_ds.zip(z_dz) {
            let si = s + α * ds;
            let zi = z + α * dz;
            barrier += (si * zi).logsafe();
        }
        barrier
    }
}

// ---------------------------------------------
// operations supported by symmetric cones only
// ---------------------------------------------

impl<T> SymmetricCone<T> for NonnegativeCone<T>
where
    T: FloatT,
{
    fn λ_inv_circ_op(&self, x: &mut [T], z: &[T]) {
        self.inv_circ_op(x, &self.λ, z);
    }

    fn mul_W(&self, _is_transpose: MatrixShape, y: &mut [T], x: &[T], α: T, β: T) {
        assert_eq!(y.len(), x.len());
        assert_eq!(y.len(), self.w.len());
        for i in 0..y.len() {
            y[i] = α * (x[i] * self.w[i]) + β * y[i];
        }
    }

    fn mul_Winv(&self, _is_transpose: MatrixShape, y: &mut [T], x: &[T], α: T, β: T) {
        assert_eq!(y.len(), x.len());
        assert_eq!(y.len(), self.w.len());
        for i in 0..y.len() {
            y[i] = α * (x[i] / self.w[i]) + β * y[i];
        }
    }
}

// ---------------------------------------------
// Jordan algebra operations for symmetric cones
// ---------------------------------------------

impl<T> JordanAlgebra<T> for NonnegativeCone<T>
where
    T: FloatT,
{
    fn circ_op(&self, x: &mut [T], y: &[T], z: &[T]) {
        let yz = y.iter().zip(z.iter());

        for (x, (y, z)) in x.iter_mut().zip(yz) {
            *x = (*y) * (*z);
        }
    }

    fn inv_circ_op(&self, x: &mut [T], y: &[T], z: &[T]) {
        let yz = y.iter().zip(z.iter());

        for (x, (y, z)) in x.iter_mut().zip(yz) {
            *x = (*z) / (*y);
        }
    }
}
