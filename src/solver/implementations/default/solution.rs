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

    fn finalize(
        &mut self,
        data: &DefaultProblemData<T>,
        variables: &DefaultVariables<T>,
        info: &DefaultInfo<T>,
    ) {
        self.status = info.status;
        self.obj_val = info.cost_primal;

        // if we have an infeasible problem, normalize
        // using κ to get an infeasibility certificate.
        // Otherwise use τ to get a solution.
        let scaleinv;
        if info.status.is_infeasible() {
            scaleinv = T::recip(variables.κ);
            self.obj_val = T::nan();
        } else {
            scaleinv = T::recip(variables.τ);
        }

        // also undo the equilibration
        let d = &data.equilibration.d;
        let (e, einv) = (&data.equilibration.e, &data.equilibration.einv);
        let cscale = data.equilibration.c;

        self.x.copy_from(&variables.x).hadamard(d).scale(scaleinv);

        if let Some(map) = data.presolver.reduce_map.as_ref() {
            //

            for (&zi, &si, &ei, &einvi, &mapi) in
                izip!(&variables.z, &variables.s, e, einv, &map.keep_index)
            {
                self.z[mapi] = zi * ei * (scaleinv / cscale);
                self.s[mapi] = si * einvi * scaleinv;
            }

            // eliminated constraints get huge slacks
            // and are assumed to be nonbinding
            let infbound = data.presolver.infbound.as_T();
            let sz = zip(&mut self.s, &mut self.z);
            zip(sz, &map.keep_logical).for_each(|((si, zi), b)| {
                if !b {
                    *si = infbound;
                    *zi = T::zero();
                }
            });
        } else {
            self.z
                .copy_from(&variables.z)
                .hadamard(e)
                .scale(scaleinv / cscale);
            self.s
                .copy_from(&variables.s)
                .hadamard(einv)
                .scale(scaleinv);
        }

        self.iterations = info.iterations;
        self.solve_time = info.solve_time;
        self.r_prim = info.res_primal;
        self.r_dual = info.res_dual;
    }
}
