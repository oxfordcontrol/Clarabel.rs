#![allow(non_snake_case)]
use super::*;
use crate::algebra::*;
use crate::solver::core::traits::Residuals;

// ---------------
// Residuals type for default problem format
// ---------------

/// Standard-form solver type implementing the [`Residuals`](crate::solver::core::traits::Residuals) trait

pub struct DefaultResiduals<T> {
    // the main KKT residuals
    pub rx: Vec<T>,
    pub rz: Vec<T>,
    pub rτ: T,

    // partial residuals for infeasibility checks
    pub rx_inf: Vec<T>,
    pub rz_inf: Vec<T>,

    // various inner products.
    // NB: these are invariant w.r.t equilibration
    pub dot_qx: T,
    pub dot_bz: T,
    pub dot_sz: T,
    pub dot_xPx: T,

    // the product Px by itself. Required for infeasibilty checks
    pub Px: Vec<T>,
}

impl<T> DefaultResiduals<T>
where
    T: FloatT,
{
    pub fn new(n: usize, m: usize) -> Self {
        let rx = vec![T::zero(); n];
        let rz = vec![T::zero(); m];
        let rτ = T::one();

        let rx_inf = vec![T::zero(); n];
        let rz_inf = vec![T::zero(); m];

        let Px = vec![T::zero(); n];

        Self {
            rx,
            rz,
            rτ,
            rx_inf,
            rz_inf,
            Px,
            dot_qx: T::zero(),
            dot_bz: T::zero(),
            dot_sz: T::zero(),
            dot_xPx: T::zero(),
        }
    }
}

impl<T> Residuals<T> for DefaultResiduals<T>
where
    T: FloatT,
{
    type D = DefaultProblemData<T>;
    type V = DefaultVariables<T>;

    fn update(&mut self, variables: &DefaultVariables<T>, data: &DefaultProblemData<T>) {
        // various products used multiple times
        let qx = data.q.dot(&variables.x);
        let bz = data.b.dot(&variables.z);
        let sz = variables.s.dot(&variables.z);

        //Px = P*x, P treated as symmetric
        let symP = data.P.sym();
        symP.symv(&mut self.Px, &variables.x, T::one(), T::zero());

        let xPx = variables.x.dot(&self.Px);

        //partial residual calc so we can check primal/dual
        //infeasibility conditions

        //Same as:
        //rx_inf .= -data.A'* variables.z
        let At = data.A.t();
        At.gemv(&mut self.rx_inf, &variables.z, -T::one(), T::zero());

        //Same as:  residuals.rz_inf .=  data.A * variables.x + variables.s
        self.rz_inf.copy_from(&variables.s);
        let A = &data.A;
        A.gemv(&mut self.rz_inf, &variables.x, T::one(), T::one());

        //complete the residuals
        //rx = rx_inf - Px - qτ
        self.rx.waxpby(-T::one(), &self.Px, -variables.τ, &data.q);
        self.rx.axpby(T::one(), &self.rx_inf, T::one());

        // rz = rz_inf - bτ
        self.rz
            .waxpby(T::one(), &self.rz_inf, -variables.τ, &data.b);

        // τ = qz + bz + κ + xPx/τ;
        self.rτ = qx + bz + variables.κ + xPx / variables.τ;

        //save local versions
        self.dot_qx = qx;
        self.dot_bz = bz;
        self.dot_sz = sz;
        self.dot_xPx = xPx;
    }
}
