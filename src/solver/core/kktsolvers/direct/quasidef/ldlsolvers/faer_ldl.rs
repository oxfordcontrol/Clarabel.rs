#![allow(non_snake_case)]

use faer::{
    dyn_stack::{GlobalPodBuffer, PodStack},
    linalg::cholesky::ldlt_diagonal::compute::LdltRegularization,
    sparse::{
        linalg::{amd::Control, cholesky::*, SupernodalThreshold},
        SparseColMatRef, SymbolicSparseColMatRef,
    },
    Conj, Parallelism, Side,
};

use crate::algebra::*;
use crate::solver::core::kktsolvers::direct::DirectLDLSolver;
use crate::solver::core::CoreSettings;

struct FaerLDLRegularizerParams<T: FloatT> {
    regularize: bool,
    eps: T,
    delta: T,
}

impl<'a, T> FaerLDLRegularizerParams<T>
where
    T: FloatT,
{
    fn to_faer(&self, Dsigns: &'a [i8]) -> LdltRegularization<'a, T> {
        let option = if self.regularize { Some(Dsigns) } else { None };
        LdltRegularization {
            dynamic_regularization_signs: option,
            dynamic_regularization_delta: self.delta,
            dynamic_regularization_epsilon: self.eps,
        }
    }
}

pub struct FaerDirectLDLSolver<T: FloatT> {
    regularizer_params: FaerLDLRegularizerParams<T>,
    Dsigns: Vec<i8>,
    symbolic_cholesky: SymbolicCholesky<usize>,
    ld_vals: Vec<T>,
    work: GlobalPodBuffer,
}

impl<T> FaerDirectLDLSolver<T>
where
    T: FloatT,
{
    pub fn new(KKT: &CscMatrix<T>, Dsigns: &[i8], settings: &CoreSettings<T>) -> Self {
        assert!(KKT.is_square(), "KKT matrix is not square");

        // -----------------------------
        // cholesky and AMD configuration
        // configure faer AMD to match behaviour used in QDLDL
        let amd_dense_scale = 1.5;
        let mut amd_params = Control {
            ..Default::default()
        };
        amd_params.dense *= amd_dense_scale;

        let supernodal_flop_ratio_threshold = SupernodalThreshold::AUTO;
        let cholesky_params = CholeskySymbolicParams {
            supernodal_flop_ratio_threshold,
            amd_params,
            ..Default::default()
        };
        // -----------------------------

        let symbKKT =
            SymbolicSparseColMatRef::new_checked(KKT.n, KKT.n, &KKT.colptr, None, &KKT.rowval);

        let symbolic_cholesky =
            factorize_symbolic_cholesky(symbKKT, Side::Upper, cholesky_params).unwrap();

        let ld_vals = vec![T::zero(); symbolic_cholesky.len_values()];

        //PJG: I don't really want to take a copy of the Dsigns here, but
        //it's not obvious how to avoid it given lifetime constraints
        let Dsigns = Dsigns.to_vec();

        let regularizer_params = FaerLDLRegularizerParams {
            regularize: settings.dynamic_regularization_enable,
            delta: settings.dynamic_regularization_delta,
            eps: settings.dynamic_regularization_eps,
        };

        let work = GlobalPodBuffer::new(
            symbolic_cholesky
                .factorize_numeric_ldlt_req::<T>(true, Parallelism::None)
                .unwrap(),
        );

        Self {
            regularizer_params,
            Dsigns,
            symbolic_cholesky,
            ld_vals,
            work,
        }
    }
}

impl<T> DirectLDLSolver<T> for FaerDirectLDLSolver<T>
where
    T: FloatT + faer::Entity + faer::SimpleEntity,
{
    fn update_values(&mut self, _index: &[usize], _values: &[T]) {
        // does not maintain internal copy of KKT, so nothing to do
    }

    fn scale_values(&mut self, _index: &[usize], _scale: T) {
        // does not maintain internal copy of KKT, so nothing to do
    }

    fn offset_values(&mut self, _index: &[usize], _offset: T, _signs: &[i8]) {
        // does not maintain internal copy of KKT, so nothing to do
    }

    fn solve(&mut self, x: &mut [T], b: &[T]) {
        // NB: faer solves in place
        x.copy_from(b);

        let ldlt = LdltRef::new(&self.symbolic_cholesky, self.ld_vals.as_slice());

        let rhs = faer::mat::from_column_major_slice_mut::<T>(&mut x[0..], b.len(), 1);

        ldlt.solve_in_place_with_conj(
            Conj::No,
            rhs,
            Parallelism::None,
            PodStack::new(&mut self.work),
        );
    }

    fn refactor(&mut self, kkt: &CscMatrix<T>) -> bool {
        let symbKKT =
            SymbolicSparseColMatRef::new_checked(kkt.n, kkt.n, &kkt.colptr, None, &kkt.rowval);

        let a: SparseColMatRef<usize, T> = SparseColMatRef::new(symbKKT, kkt.nzval.as_slice());

        let regularizer = self.regularizer_params.to_faer(&self.Dsigns);

        self.symbolic_cholesky.factorize_numeric_ldlt(
            self.ld_vals.as_mut_slice(),
            a,
            Side::Upper,
            regularizer,
            Parallelism::None,
            PodStack::new(&mut self.work),
        );

        true // assume success for experimental purposes
    }

    fn required_matrix_shape() -> MatrixTriangle {
        MatrixTriangle::Triu
    }
}
