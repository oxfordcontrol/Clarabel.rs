use super::Cone;
use clarabel_algebra::*;

// -------------------------------------
// Nonnegative Cone
// -------------------------------------

pub struct NonnegativeCone<T: FloatT = f64> {
    dim: usize,
    w: Vec<T>,
    λ: Vec<T>,
}

impl<T: FloatT> NonnegativeCone<T> {
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

    fn rectify_equilibration(&self, δ: &mut [T], e: &[T]) -> bool {
        δ.copy_from(e);
        false
    }

    fn WtW_is_diagonal(&self) -> bool {
        true
    }

    fn update_scaling(&mut self, s: &[T], z: &[T]) {
        let λw = self.λ.iter_mut().zip(self.w.iter_mut());
        let sz = s.iter().zip(z.iter());

        for ((λ, w), (s, z)) in λw.zip(sz) {
            *λ = T::sqrt((*s) * (*z));
            *w = T::sqrt((*s) / (*z));
        }
    }

    fn set_identity_scaling(&mut self) {
        self.w.fill(T::one());
    }

    fn λ_circ_λ(&self, x: &mut [T]) {
        assert_eq!(self.λ.len(), x.len());
        for (xi, &λi) in x.iter_mut().zip(self.λ.iter()) {
            *xi = λi * λi;
        }
    }

    fn circ_op(&self, x: &mut [T], y: &[T], z: &[T]) {
        let yz = y.iter().zip(z.iter());

        for (x, (y, z)) in x.iter_mut().zip(yz) {
            *x = (*y) * (*z);
        }
    }

    fn λ_inv_circ_op(&self, x: &mut [T], z: &[T]) {
        self.inv_circ_op(x, &self.λ, z)
    }

    fn inv_circ_op(&self, x: &mut [T], y: &[T], z: &[T]) {
        let yz = y.iter().zip(z.iter());

        for (x, (y, z)) in x.iter_mut().zip(yz) {
            *x = (*z) / (*y);
        }
    }

    fn shift_to_cone(&self, z: &mut [T]) {
        let α = z.minimum();
        if α < T::epsilon() {
            //done in two stages since otherwise (1-α) = -α for
            //large α, which makes z exactly 0. (or worse, -0.0 )
            self.add_scaled_e(z, -α);
            self.add_scaled_e(z, T::one());
        }
    }

    fn get_WtW_block(&self, WtWblock: &mut [T]) {
        assert_eq!(self.w.len(), WtWblock.len());
        for (blki, &wi) in WtWblock.iter_mut().zip(self.w.iter()) {
            *blki = wi * wi;
        }
    }

    fn gemv_W(&self, _is_transpose: MatrixShape, x: &[T], y: &mut [T], α: T, β: T) {
        assert_eq!(y.len(), x.len());
        assert_eq!(y.len(), self.w.len());
        for i in 0..y.len() {
            y[i] = α * (x[i] * self.w[i]) + β * y[i];
        }
    }

    fn gemv_Winv(&self, _is_transpose: MatrixShape, x: &[T], y: &mut [T], α: T, β: T) {
        assert_eq!(y.len(), x.len());
        assert_eq!(y.len(), self.w.len());
        for i in 0..y.len() {
            y[i] = α * (x[i] / self.w[i]) + β * y[i];
        }
    }

    fn add_scaled_e(&self, x: &mut [T], α: T) {
        x.translate(α);
    }

    fn step_length(&self, dz: &[T], ds: &[T], z: &[T], s: &[T]) -> (T, T) {
        let mut αz = T::max_value();
        let mut αs = T::max_value();

        assert_eq!(z.len(), s.len());
        assert_eq!(dz.len(), z.len());
        assert_eq!(ds.len(), s.len());

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
}
