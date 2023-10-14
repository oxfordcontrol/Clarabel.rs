#![allow(non_snake_case)]
#![allow(unused_imports)]
#![allow(dead_code)]
#![allow(unused_variables)]

use crate::algebra::*;
use crate::solver::core::kktsolvers::direct::DirectLDLSolver;
use crate::solver::core::CoreSettings;
use dyn_stack::{GlobalPodBuffer, PodStack};
use faer_core::{ComplexField, Conj, Entity, MatMut, Parallelism};
use faer_sparse_experimental::{
    cholesky::*, Side, SliceGroup, SliceGroupMut, SparseColMatRef, SymbolicSparseColMatRef,
};

pub struct FaerDirectLDLSolver<T> {
    symbolic: SymbolicCholesky<i64>,
    ld_vals: Vec<T>,
    work: GlobalPodBuffer,
    Dsigns: Vec<i8>,
}

impl<T> FaerDirectLDLSolver<T>
where
    T: FloatT,
{
    pub fn new(KKT: &CscMatrix<T>, Dsigns: &[i8], settings: &CoreSettings<T>) -> Self {
        assert!(KKT.is_square(), "KKT matrix is not square");

        let n = KKT.ncols();

        // PJG: faer expects i64 for colptr and rowval, and doesn't like
        // usize as we use internally.   This is because the faer bounds
        // want the Neg trait for row and column indices.
        let col_ptr: Vec<i64> = KKT.colptr.iter().map(|&x| x as i64).collect();
        let row_ind: Vec<i64> = KKT.rowval.iter().map(|&x| x as i64).collect();

        let symbmat = SymbolicSparseColMatRef::new_checked(n, n, &col_ptr, None, &row_ind);
        let symbolic =
            factorize_symbolic(symbmat, Side::Upper, CholeskySymbolicParams::default()).unwrap();
        let ld_vals = vec![T::zero(); symbolic.len_values()];

        //PJG: I don't really want to take a copy of the Dsigns here, but
        //it's not obvious how to avoid it given lifetime constraints
        let Dsigns = Dsigns.to_vec();

        let regularizer: LdltRegularization<T> = LdltRegularization {
            dynamic_regularization_signs: Some(&Dsigns.to_vec()),
            dynamic_regularization_delta: settings.dynamic_regularization_delta,
            dynamic_regularization_epsilon: settings.dynamic_regularization_eps,
        };

        let scratch_memory = symbolic
            .factorize_numeric_ldlt_req::<T>(false, Parallelism::None)
            .unwrap()
            .try_or(symbolic.dense_solve_in_place_req::<T>(1).unwrap())
            .unwrap();

        let work = GlobalPodBuffer::try_new(scratch_memory).unwrap();

        Self {
            symbolic,
            ld_vals,
            work,
            Dsigns,
        }
    }
}

impl<T> DirectLDLSolver<T> for FaerDirectLDLSolver<T>
where
    T: FloatT,
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

        // PJG: problem here - not clear what bounds are required on T
        let tmp = SliceGroup::<T>::new(&self.ld_vals);

        let ldlt = LdltRef::new(&self.symbolic, tmp);
        // let rhs = MatMut::from_column_major_slice(x, x.len(), 1);
        // ldlt.dense_solve_in_place_with_conj(
        //     rhs,
        //     Conj::No,
        //     Parallelism::None,
        //     PodStack::new(&mut self.work),
        // );
    }

    fn refactor(&mut self, kkt: &CscMatrix<T>) -> bool {
        //let kkt_values = &kkt.nzval;

        //let a = SparseColMatRef::new(self.symbmat, SliceGroup::<T>::new(kkt_values));

        //PJG: I don't want to recreate the regularizer here since I don't
        //have direct access to the settings when refactoring.   Would be
        //better if the regularizer was created once and then storeds with
        //the struct.
        // let regularizer = LdltRegularization {
        //     dynamic_regularization_signs: Some(&self.Dsigns),
        //     dynamic_regularization_delta: 1e-8,
        //     dynamic_regularization_epsilon: 1e-13,
        // };

        // self.symbolic.factorize_numeric_ldlt(
        //     SliceGroupMut::<T>::new(&mut self.ld_vals),
        //     a,
        //     Side::Upper,
        //     regularizer,
        //     Parallelism::None,
        //     PodStack::new(&mut self.work),
        // );

        true // assume success for experimental purposes
    }

    fn required_matrix_shape() -> MatrixTriangle {
        MatrixTriangle::Triu
    }
}
