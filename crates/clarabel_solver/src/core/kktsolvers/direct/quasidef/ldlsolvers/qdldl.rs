#![allow(non_snake_case)]
use super::super::DirectLDLSolver;
use crate::core::CoreSettings;
use clarabel_algebra::*;
use clarabel_qdldl::*;

pub struct QDLDLDirectLDLSolver<T> {

    //KKT matrix and its QDLDL factorization
    factors: QDLDLFactorisation<T>,

}

impl<T: FloatT> QDLDLDirectLDLSolver<T> {
    pub fn new(KKT: &CscMatrix<T>, Dsigns: &[i8], settings: &CoreSettings<T>) -> Self {
        let dim = KKT.nrows();

        assert!(dim == KKT.ncols(), "KKT matrix is not square");

        //construct the LDL solver settings
        let opts = QDLDLSettingsBuilder::default()
            .logical(true) //allocate memory only on init
            .Dsigns(Dsigns.to_vec())
            .regularize_enable(true)
            .regularize_eps(settings.dynamic_regularization_eps)
            .regularize_delta(settings.dynamic_regularization_delta)
            .build()
            .unwrap();

        let factors = QDLDLFactorisation::<T>::new(KKT, Some(opts));

        Self {factors}
    }
}

impl<T: FloatT> DirectLDLSolver<T> for QDLDLDirectLDLSolver<T> {

    fn update_values(&mut self, index: &[usize], values: &[T]) {

        //Update values that are stored within
        //the reordered copy held internally by QDLDL.
        self.factors.update_values(index, values);
    }

    fn scale_values(&mut self, index: &[usize], scale: T) {

        self.factors.scale_values(index, scale);
    }

    fn offset_values(&mut self, index: &[usize], offset: T, signs: &[i8]) {

        self.factors.offset_values(index, offset, signs);
    }

    fn solve(&mut self, x: &mut [T], b: &[T]) {
        // NB: QDLDL solves in place
        x.copy_from(b);
        self.factors.solve(x);
    }

    fn refactor(&mut self, _kkt: &CscMatrix<T>) {

        //QDLDL has maintained its own version of the permuted
        //KKT matrix through custom update/scale/offset methods,
        //so we ignore the KKT matrix provided by the caller
        self.factors.refactor();
    }
}
