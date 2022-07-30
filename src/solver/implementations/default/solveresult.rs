use std::time::Duration;

use super::*;
use crate::{
    algebra::*,
    solver::core::{traits::Solution, SolverStatus},
};

/// Standard-form solver type implementing the [Solution](crate::solver::core::traits::Solution) trait

pub struct DefaultSolution<T> {
    pub x: Vec<T>,
    pub z: Vec<T>,
    pub s: Vec<T>,
    pub status: SolverStatus,
    pub obj_val: T,
    pub solve_time: std::time::Duration,
    pub iterations: u32,
    pub r_prim: T,
    pub r_dual: T,
}

impl<T> DefaultSolution<T>
where
    T: FloatT,
{
    pub fn new(m: usize, n: usize) -> Self {
        let x = vec![T::zero(); n];
        let z = vec![T::zero(); m];
        let s = vec![T::zero(); m];

        Self {
            x,
            z,
            s,
            status: SolverStatus::Unsolved,
            obj_val: T::nan(),
            solve_time: Duration::ZERO,
            iterations: 0,
            r_prim: T::nan(),
            r_dual: T::nan(),
        }
    }
}

impl<T> Solution<T> for DefaultSolution<T>
where
    T: FloatT,
{
    type D = DefaultProblemData<T>;
    type V = DefaultVariables<T>;
    type I = DefaultInfo<T>;

    fn finalize(
        &mut self,
        data: &DefaultProblemData<T>,
        variables: &DefaultVariables<T>,
        info: &DefaultInfo<T>,
    ) {
        self.status = info.status.clone();
        self.obj_val = info.cost_primal;

        //copy internal variables and undo homogenization
        self.x.copy_from(&variables.x);
        self.z.copy_from(&variables.z);
        self.s.copy_from(&variables.s);

        // if we have an infeasible problem, normalize
        // using κ to get an infeasibility certificate.
        // Otherwise use τ to get a solution.
        let scaleinv;
        if info.status == SolverStatus::PrimalInfeasible
            || info.status == SolverStatus::DualInfeasible
        {
            scaleinv = T::recip(variables.κ);
            self.obj_val = T::nan();
        } else {
            scaleinv = T::recip(variables.τ);
        }

        self.x.scale(scaleinv);
        self.z.scale(scaleinv);
        self.s.scale(scaleinv);

        // undo the equilibration
        let d = &data.equilibration.d;
        let (e, einv) = (&data.equilibration.e, &data.equilibration.einv);
        let cscale = data.equilibration.c;

        self.x.hadamard(d);
        self.z.hadamard(e);
        self.z.scale(T::recip(cscale));
        self.s.hadamard(einv);

        self.iterations = info.iterations;
        self.solve_time = info.solve_time;
        self.r_prim = info.res_primal;
        self.r_dual = info.res_dual;
    }
}
