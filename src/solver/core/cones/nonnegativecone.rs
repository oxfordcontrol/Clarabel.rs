use super::*;
use crate::algebra::*;
use itertools::izip;
use std::iter::zip;

// -------------------------------------
// Nonnegative Cone
// -------------------------------------

pub struct NonnegativeCone<T> {
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
    fn degree(&self) -> usize {
        self.dim
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

    fn margins(&mut self, z: &mut [T], _pd: PrimalOrDualCone) -> (T, T) {
        let α = z.minimum();
        let β = z.iter().fold(T::zero(), |β, zi| β + T::max(zi.clone(), T::zero()));
        (α, β)
    }

    fn scaled_unit_shift(&self, z: &mut [T], α: T, _pd: PrimalOrDualCone) {
        z.translate(α);
    }

    fn unit_initialization(&self, z: &mut [T], s: &mut [T]) {
        z.set(T::one());
        s.set(T::one());
    }

    fn set_identity_scaling(&mut self) {
        self.w.set(T::one());
    }

    fn update_scaling(
        &mut self,
        s: &[T],
        z: &[T],
        _μ: T,
        _scaling_strategy: ScalingStrategy,
    ) -> bool {
        for (λ, w, s, z) in izip!(&mut self.λ, &mut self.w, s, z) {
            *λ = T::sqrt(s.clone() * z.clone());
            *w = T::sqrt(s.clone() / z.clone());
        }

        true
    }

    fn Hs_is_diagonal(&self) -> bool {
        true
    }

    fn get_Hs(&self, Hsblock: &mut [T]) {
        assert_eq!(self.w.len(), Hsblock.len());
        for (blki, wi) in zip(Hsblock, &self.w) {
            *blki = wi.clone() * wi.clone();
        }
    }

    fn mul_Hs(&mut self, y: &mut [T], x: &[T], _work: &mut [T]) {
        //NB : seemingly sensitive to order of multiplication
        for (yi, (wi, xi)) in y.iter_mut().zip(self.w.iter().zip(x)) {
            *yi = wi.clone() * (wi.clone() * xi.clone())
        }
    }

    fn affine_ds(&self, ds: &mut [T], _s: &[T]) {
        assert_eq!(self.λ.len(), ds.len());
        for (dsi, λi) in zip(ds, &self.λ) {
            *dsi = λi.clone() * λi.clone();
        }
    }

    fn combined_ds_shift(&mut self, dz: &mut [T], step_z: &mut [T], step_s: &mut [T], σμ: T) {
        //PJG: could be done faster for nonnegatives?
        self._combined_ds_shift_symmetric(dz, step_z, step_s, σμ);
    }

    fn Δs_from_Δz_offset(&mut self, out: &mut [T], ds: &[T], _work: &mut [T], z: &[T]) {
        for (outi, (dsi, zi)) in zip(out, zip(ds, z)) {
            *outi = dsi.clone() / zi.clone();
        }
    }

    fn step_length(
        &mut self,
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

        let mut αz = αmax.clone();
        let mut αs = αmax;

        for i in 0..z.len() {
            if dz[i] < T::zero() {
                αz = T::min(αz, -z[i].clone() / dz[i].clone());
            }
            if ds[i] < T::zero() {
                αs = T::min(αs, -s[i].clone() / ds[i].clone());
            }
        }
        (αz, αs)
    }

    fn compute_barrier(&mut self, z: &[T], s: &[T], dz: &[T], ds: &[T], α: T) -> T {
        assert_eq!(z.len(), s.len());
        assert_eq!(dz.len(), z.len());
        assert_eq!(ds.len(), s.len());
        let mut barrier = T::zero();
        for (s, ds, z, dz) in izip!(s, ds, z, dz) {
            let si = s.clone() + α.clone() * ds.clone();
            let zi = z.clone() + α.clone() * dz.clone();
            barrier -= (si * zi).logsafe();
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
    fn λ_inv_circ_op(&mut self, x: &mut [T], z: &[T]) {
        _inv_circ_op(x, &self.λ, z);
    }

    fn mul_W(&mut self, _is_transpose: MatrixShape, y: &mut [T], x: &[T], α: T, β: T) {
        assert_eq!(y.len(), x.len());
        assert_eq!(y.len(), self.w.len());
        for i in 0..y.len() {
            y[i] = α.clone() * (x[i].clone() * self.w[i].clone()) + β.clone() * y[i].clone();
        }
    }

    fn mul_Winv(&mut self, _is_transpose: MatrixShape, y: &mut [T], x: &[T], α: T, β: T) {
        assert_eq!(y.len(), x.len());
        assert_eq!(y.len(), self.w.len());
        for i in 0..y.len() {
            y[i] = α.clone() * (x[i].clone() / self.w[i].clone()) + β.clone() * y[i].clone();
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
    fn circ_op(&mut self, x: &mut [T], y: &[T], z: &[T]) {
        _circ_op(x, y, z);
    }

    fn inv_circ_op(&mut self, x: &mut [T], y: &[T], z: &[T]) {
        _inv_circ_op(x, y, z);
    }
}

// circ ops don't use self for this cone, so put the actual
// implementations outside so that they can be called by
// other functions with entering borrow check hell
fn _circ_op<T>(x: &mut [T], y: &[T], z: &[T])
where
    T: FloatT,
{
    for (x, (y, z)) in zip(x, zip(y, z)) {
        *x = y.clone() * z.clone();
    }
}

fn _inv_circ_op<T>(x: &mut [T], y: &[T], z: &[T])
where
    T: FloatT,
{
    for (x, (y, z)) in zip(x, zip(y, z)) {
        *x = z.clone() / y.clone();
    }
}
