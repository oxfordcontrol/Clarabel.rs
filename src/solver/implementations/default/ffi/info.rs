use crate::{
    algebra::*,
    solver::{core::ffi::*, DefaultInfo},
};

/// FFI interface for [`DefaultInfo`](crate::solver::default::DefaultInfo)
#[repr(C)]
#[allow(missing_docs)]
pub struct DefaultInfoFFI<T> {
    pub mu: T,
    pub sigma: T,
    pub step_length: T,
    pub iterations: u32,
    pub cost_primal: T,
    pub cost_dual: T,
    pub res_primal: T,
    pub res_dual: T,
    pub res_primal_inf: T,
    pub res_dual_inf: T,
    pub gap_abs: T,
    pub gap_rel: T,
    pub ktratio: T,

    pub solve_time: f64,
    pub status: SolverStatusFFI,
    pub linsolver: LinearSolverInfoFFI,
}

impl<T: FloatT> From<DefaultInfo<T>> for DefaultInfoFFI<T> {
    fn from(info: DefaultInfo<T>) -> Self {
        Self {
            mu: info.mu,
            sigma: info.sigma,
            step_length: info.step_length,
            iterations: info.iterations,
            cost_primal: info.cost_primal,
            cost_dual: info.cost_dual,
            res_primal: info.res_primal,
            res_dual: info.res_dual,
            res_primal_inf: info.res_primal_inf,
            res_dual_inf: info.res_dual_inf,
            gap_abs: info.gap_abs,
            gap_rel: info.gap_rel,
            ktratio: info.ktratio,
            solve_time: info.solve_time,
            status: info.status,
            linsolver: info.linsolver.into(),
        }
    }
}

#[test]
fn test_info_ffi() {
    use super::*;

    let info = DefaultInfo::<f64> {
        ktratio: 2.0,
        ..Default::default()
    };
    let info_ffi: DefaultInfoFFI<f64> = info.clone().into();

    assert_eq!(info_ffi.ktratio, info.ktratio);
}
