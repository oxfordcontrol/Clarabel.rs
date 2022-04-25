#![allow(non_snake_case)]
use crate::algebra::*;
use clarabel_qdldl::*;
use crate::kktsolvers::direct_quasidefinite::DirectLDLSolver;
use crate::Settings;

pub struct QDLDLDirectLDLSolver<T:FloatT>{

    //KKT matrix and its QDLDL factorization
    KKT: CscMatrix<T>,
    factors: QDLDLFactorisation<T>,

    //internal workspace for IR scheme
    work: Vec<T>
}

impl<T:FloatT> QDLDLDirectLDLSolver<T>{

    pub fn new(KKT: CscMatrix<T>, Dsigns: Vec<i8>, settings: &Settings<T>) -> Self {

        let dim = KKT.nrows();

        assert!(dim == KKT.ncols(), "KKT matrix is not square");

        //construct the LDL solver settings
        let opts = QDLDLSettingsBuilder::default()
                .logical(true)          //allocate memory only on init
                .Dsigns(Dsigns)
                .regularize(true)
                .regularize_eps(settings.dynamic_regularization_eps)
                .regularize_delta(settings.dynamic_regularization_delta)
                .build().unwrap();

        let factors = QDLDLFactorisation::<T>::new(&KKT, Some(opts));

        let work = vec![T::zero(); dim];

        Self{KKT,factors,work}


    }

}

impl<T: FloatT> DirectLDLSolver<T> for QDLDLDirectLDLSolver<T> {
    fn update_values(&mut self, _index: &[usize], _values: &[T]){unimplemented!();}
    fn offset_values(&mut self, _index: &[usize], _value: T){unimplemented!();}
    fn solve(&mut self, _x: &mut [T],_b: &[T]){unimplemented!();}
    fn refactor(&mut self){unimplemented!();}
}
