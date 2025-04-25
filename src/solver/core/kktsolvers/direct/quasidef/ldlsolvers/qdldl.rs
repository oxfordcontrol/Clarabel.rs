#![allow(non_snake_case)]
use crate::algebra::*;
use crate::qdldl::*;
use crate::solver::core::kktsolvers::direct::{DirectLDLSolver, DirectLDLSolverReqs};
use crate::solver::core::kktsolvers::HasLinearSolverInfo;
use crate::solver::core::kktsolvers::LinearSolverInfo;
use crate::solver::core::CoreSettings;

pub struct QDLDLDirectLDLSolver<T> {
    //KKT matrix and its QDLDL factorization
    factors: QDLDLFactorisation<T>,
}

impl<T> QDLDLDirectLDLSolver<T>
where
    T: FloatT,
{
    pub fn new(
        KKT: &CscMatrix<T>,
        Dsigns: &[i8],
        settings: &CoreSettings<T>,
        perm: Option<Vec<usize>>,
    ) -> Self {
        assert!(KKT.is_square(), "KKT matrix is not square");

        // occasionally we find that the default AMD parameters give a bad ordering, particularly
        // for some big matrices.  In particular, KKT conditions for QPs are sometimes worse
        // than their SOC counterparts for very large problems.   This is because the SOC form
        // is artificially "big", with extra rows, so the dense row threshold is effectively a
        // different value.   We fix a bit more generous AMD_DENSE here, which should perhaps
        // be user-settable.

        //make a logical factorization to determine memory allocations

        let mut opts = QDLDLSettingsBuilder::default()
            .logical(true) //allocate memory only on init
            .Dsigns(Dsigns.to_vec())
            .regularize_enable(true)
            .regularize_eps(settings.dynamic_regularization_eps)
            .regularize_delta(settings.dynamic_regularization_delta)
            .amd_dense_scale(1.5)
            .build()
            .unwrap();
        opts.perm = perm;

        let factors = QDLDLFactorisation::<T>::new(KKT, Some(opts)).unwrap();

        Self { factors }
    }
}

impl<T> DirectLDLSolverReqs for QDLDLDirectLDLSolver<T>
where
    T: FloatT,
{
    fn required_matrix_shape() -> MatrixTriangle {
        MatrixTriangle::Triu
    }
}

impl<T> HasLinearSolverInfo for QDLDLDirectLDLSolver<T>
where
    T: FloatT,
{
    fn linear_solver_info(&self) -> LinearSolverInfo {
        LinearSolverInfo {
            name: "qdldl".to_string(),
            threads: 1,
            direct: true,
            nnzA: self.factors.nnzA(),
            nnzL: self.factors.nnzL(),
        }
    }
}

impl<T> DirectLDLSolver<T> for QDLDLDirectLDLSolver<T>
where
    T: FloatT,
{
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

    fn solve(&mut self, _kkt: &CscMatrix<T>, x: &mut [T], b: &mut [T]) {
        // NB: QDLDL solves in place
        x.copy_from(b);
        self.factors.solve(x);
    }

    fn refactor(&mut self, _kkt: &CscMatrix<T>) -> bool {
        //QDLDL has maintained its own version of the permuted
        //KKT matrix through custom update/scale/offset methods,
        //so we ignore the KKT matrix provided by the caller
        self.factors.refactor().unwrap();
        self.factors.Dinv.is_finite()
    }
}
