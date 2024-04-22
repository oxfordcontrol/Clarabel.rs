use super::*;
use crate::{
    algebra::*,
    solver::core::{traits::Solution, SolverStatus},
};
use itertools::izip;
use std::iter::zip;

/// Standard-form solver type implementing the [`Solution`](crate::solver::core::traits::Solution) trait

pub struct DefaultSolution<T> {
    pub x: Vec<T>,
    pub z: Vec<T>,
    pub s: Vec<T>,
    pub status: SolverStatus,
    pub obj_val: T,
    pub obj_val_dual: T,
    pub solve_time: f64,
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
            obj_val_dual: T::nan(),
            solve_time: 0f64,
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
    type SE = DefaultSettings<T>;

    fn finalize(
        &mut self,
        data: &DefaultProblemData<T>,
        variables: &mut DefaultVariables<T>,
        info: &DefaultInfo<T>,
        settings: &DefaultSettings<T>,
    ) {
        self.status = info.status;
        let is_infeasible = info.status.is_infeasible();

        if is_infeasible {
            self.obj_val = T::nan();
            self.obj_val_dual = T::nan();
        } else {
            self.obj_val = info.cost_primal;
            self.obj_val_dual = info.cost_dual;
        }

        self.iterations = info.iterations;
        self.solve_time = info.solve_time;
        self.r_prim = info.res_primal;
        self.r_dual = info.res_dual;

        // unscale the variables to get a solution
        // to the internal problem as we solved it
        variables.unscale(data, is_infeasible);

        // unwind the chordal decomp and presolve, in the
        // reverse of the order in which they were applied
        let tmp = {
            if let Some(ref chordal_info) = data.chordal_info {
                Some(chordal_info.decomp_reverse(&variables, &data.cones, settings))
            } else {
                None
            }
        };
        let variables = tmp.as_ref().unwrap_or_else(|| variables);

        if let Some(ref presolver) = data.presolver {
            presolver.reverse_presolve(self, variables);
        } else {
            self.x.copy_from(&variables.x);
            self.z.copy_from(&variables.z);
            self.s.copy_from(&variables.s);
        }
    }
}
