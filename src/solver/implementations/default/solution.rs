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

        //PJG DEBUG.   These values should disappear
        self.z.set((1234.).as_T());
        self.s.set((1234.).as_T());

        if !data.presolver.is_reduced() {
            self.z
                .copy_from(&variables.z)
                .hadamard(e)
                .scale(scaleinv / cscale);
            self.s
                .copy_from(&variables.s)
                .hadamard(einv)
                .scale(scaleinv);
        } else {
            //PJG : temporary alloc makes implementation much easier
            //here. could also use something like e or einv as scratch
            let mut tmp = vec![T::zero(); variables.s.len()];

            let map = data.presolver.lift_map.as_ref().unwrap();
            let reduce_idx = data.presolver.reduce_idx.as_ref().unwrap();

            tmp.copy_from(&variables.z)
                .hadamard(e)
                .scale(scaleinv / cscale);
            for (vi, mapi) in tmp.iter().zip(map) {
                self.z[*mapi] = *vi;
            }

            tmp.copy_from(&variables.s).hadamard(einv).scale(scaleinv);
            for (vi, mapi) in tmp.iter().zip(map) {
                self.s[*mapi] = *vi;
            }

            // eliminated constraints get huge slacks
            // and are assumed to be nonbinding

            let infbound = data.presolver.infbound.as_T();
            self.s.iter_mut().zip(reduce_idx).for_each(|(x, b)| {
                if !b {
                    *x = infbound;
                }
            });
            self.z.iter_mut().zip(reduce_idx).for_each(|(x, b)| {
                if !b {
                    *x = T::zero();
                }
            });
        }

        self.iterations = info.iterations;
        self.solve_time = info.solve_time;
        self.r_prim = info.res_primal;
        self.r_dual = info.res_dual;
    }
}
