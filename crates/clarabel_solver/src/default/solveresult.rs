use crate::default::*;
use clarabel_algebra::*;

pub struct DefaultSolveResult<T> {
    pub x: Vec<T>,
    pub z: Vec<T>,
    pub s: Vec<T>,
    pub obj_val: T,
    pub status: SolverStatus,
    //PJG: leaving out solveinfo for now since
    //it contains a timer etc, plus not sure
    //now to initialize it since there is also
    //one held by the solver itself.   I guess
    //it will need to be cloned or something.

    //pub info: SolveInfo,
}

impl<T: FloatT> DefaultSolveResult<T> {
    pub fn new(m: usize, n: usize) -> Self {
        let x = vec![T::zero(); n];
        let z = vec![T::zero(); m];
        let s = vec![T::zero(); m];

        let obj_val = T::nan();
        let status = SolverStatus::Unsolved;
        //let info = ???

        Self {
            x,
            z,
            s,
            obj_val,
            status,
        }
    }
}

impl<T: FloatT> SolveResult<T> for DefaultSolveResult<T> {
    type D = DefaultProblemData<T>;
    type V = DefaultVariables<T>;
    type SI = DefaultSolveInfo<T>;

    fn finalize(
        &mut self,
        data: &DefaultProblemData<T>,
        variables: &DefaultVariables<T>,
        info: &DefaultSolveInfo<T>,
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

        //PJG : Leaving this out since the SolveInfo is not defined,
        //and it is also now super confusing with the type names of
        //the internal info object.
        // self.info.r_prim 	   = info.res_primal;
        // self.info.r_dual 	   = info.res_dual;
        // self.info.iter	   = info.iterations;
        // self.info.solve_time = info.solve_time;
    }
}
