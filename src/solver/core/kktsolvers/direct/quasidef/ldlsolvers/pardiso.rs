#![allow(non_snake_case)]
use crate::algebra::*;
use crate::solver::core::{
    kktsolvers::{
        direct::{DirectLDLSolver, DirectLDLSolverReqs},
        HasLinearSolverInfo, LinearSolverInfo,
    },
    CoreSettings,
};
use pardiso_wrapper::*;

// 32 bit indices for pardiso
struct PardisoMatrixIndices32 {
    pub colptr32: Vec<i32>,
    pub rowval32: Vec<i32>,
}

impl PardisoMatrixIndices32 {
    pub fn new(KKT: &CscMatrix<f64>) -> Self {
        let colptr32 = KKT
            .colptr
            .iter()
            .map(|&x| (x + 1) as i32)
            .collect::<Vec<i32>>();
        let rowval32 = KKT
            .rowval
            .iter()
            .map(|&x| (x + 1) as i32)
            .collect::<Vec<i32>>();
        Self { colptr32, rowval32 }
    }
}

pub struct PardisoDirectLDLSolver<P>
where
    P: PardisoInterface,
{
    ps: P,
    nnzA: usize,

    // Panua wants 32 bit CSC indices, 1 indexed
    // implemented as Option since I want to keep
    // to door open for a 64 bit 0-indexed version
    // of pardiso to pass matrix indices through
    // directly from the CscMatrix
    index32: Option<PardisoMatrixIndices32>,
}

// For a 64 bit version of pardiso, this trait would need to pass through
// either the indexed data or the CscMatrix values directly, and would
// need to be made generic over the index type.
trait PardisoDirectLDLSolverInterface {
    fn pardiso_kkt<'a>(
        &self,
        KKT: &'a CscMatrix<f64>,
        index32: &'a Option<PardisoMatrixIndices32>,
    ) -> (&'a [f64], &'a [i32], &'a [i32]);
    fn initialize(
        &mut self,
        KKT: &CscMatrix<f64>,
        Dsigns: &[i8],
        settings: &CoreSettings<f64>,
        perm: Option<Vec<usize>>,
    );
}

impl<P> PardisoDirectLDLSolverInterface for PardisoDirectLDLSolver<P>
where
    P: PardisoInterface,
{
    fn pardiso_kkt<'a>(
        &self,
        KKT: &'a CscMatrix<f64>,
        index32: &'a Option<PardisoMatrixIndices32>,
    ) -> (&'a [f64], &'a [i32], &'a [i32]) {
        // for now, assume that the option is always Some and we
        // never pass through the CscMatrix indices directly
        let index32 = index32.as_ref().unwrap();
        (&KKT.nzval, &index32.rowval32, &index32.colptr32)
    }

    fn initialize(
        &mut self,
        kkt: &CscMatrix<f64>,
        _Dsigns: &[i8],
        _settings: &CoreSettings<f64>,
        _perm: Option<Vec<usize>>,
    ) {
        // NB: ignore Dsigns here because pardiso doesn't
        // use information about the expected signs

        // perform logical factor

        let (nzvals, rowvals, colptrs) = self.pardiso_kkt(kkt, &self.index32);
        let ps = &mut self.ps;

        ps.set_matrix_type(MatrixType::RealSymmetricIndefinite);
        ps.pardisoinit().unwrap();

        // sets pardiso to solve the transposed system since we are supplying
        // CSC data and it expects CSR data.   I don't think this matters for
        // a symmetric system, but check here first for failed solves
        ps.set_iparm(11, 1);

        ps.set_phase(Phase::Analysis);

        assert_pardiso_const_rhs_config(ps);

        ps.pardiso(
            nzvals,
            colptrs,
            rowvals,
            &mut [],
            &mut [],
            kkt.n as i32,
            1_i32,
        )
        .unwrap();
    }
}

#[cfg(feature = "pardiso-mkl")]
pub(crate) type MKLPardisoDirectLDLSolver = PardisoDirectLDLSolver<MKLPardisoSolver>;

#[cfg(feature = "pardiso-mkl")]
impl MKLPardisoDirectLDLSolver {
    pub fn new(
        KKT: &CscMatrix<f64>,
        Dsigns: &[i8],
        settings: &CoreSettings<f64>,
        perm: Option<Vec<usize>>,
    ) -> Self {
        assert!(KKT.is_square(), "KKT matrix is not square");

        assert!(
            MKLPardisoSolver::is_available(),
            "MKL Pardiso is not available"
        );

        let ps = MKLPardisoSolver::new().unwrap();
        let nnzA = KKT.nnz();
        let index32 = Some(PardisoMatrixIndices32::new(KKT));

        let mut solver = Self { ps, nnzA, index32 };

        solver.initialize(KKT, Dsigns, settings, perm);
        let nthreads = settings.max_threads as i32;

        if nthreads > 0 {
            // manually set if not configured as auto select
            solver.ps.set_num_threads(nthreads).unwrap();
        }

        solver
    }
}

#[cfg(feature = "pardiso-panua")]
pub(crate) type PanuaPardisoDirectLDLSolver = PardisoDirectLDLSolver<PanuaPardisoSolver>;

#[cfg(feature = "pardiso-panua")]
impl PanuaPardisoDirectLDLSolver {
    pub fn new(
        KKT: &CscMatrix<f64>,
        Dsigns: &[i8],
        settings: &CoreSettings<f64>,
        perm: Option<Vec<usize>>,
    ) -> Self {
        assert!(KKT.is_square(), "KKT matrix is not square");

        assert!(
            PanuaPardisoSolver::is_available(),
            "Panua Pardiso is not available"
        );

        let mut ps = PanuaPardisoSolver::new().unwrap();
        let nnzA = KKT.nnz();
        let index32 = Some(PardisoMatrixIndices32::new(KKT));

        // disable internal iterative refinement (available in Panua only)
        ps.set_iparm(7, -99);

        let mut solver = Self { ps, nnzA, index32 };

        solver.initialize(KKT, Dsigns, settings, perm);

        // Note : Panua doesn't support setting the number of threads
        // Always reads instead from ENV["OMP_NUM_THREADS"] before loading

        solver
    }
}

impl<P> DirectLDLSolverReqs for PardisoDirectLDLSolver<P>
where
    P: PardisoInterface,
{
    fn required_matrix_shape() -> MatrixTriangle {
        MatrixTriangle::Tril
    }
}

impl<P> HasLinearSolverInfo for PardisoDirectLDLSolver<P>
where
    P: PardisoInterface,
{
    fn linear_solver_info(&self) -> LinearSolverInfo {
        LinearSolverInfo {
            name: self.ps.name().to_string(),
            threads: self.ps.get_num_threads().unwrap() as usize,
            direct: true,
            nnzA: self.nnzA,
            nnzL: 0, // TODO: Implement this
        }
    }
}

impl<P> DirectLDLSolver<f64> for PardisoDirectLDLSolver<P>
where
    P: PardisoInterface,
{
    fn update_values(&mut self, _index: &[usize], _values: &[f64]) {
        //no-op.  Will just use KKT matrix as passed to refactor
    }

    fn scale_values(&mut self, _index: &[usize], _scale: f64) {
        //no-op.  Will just use KKT matrix as passed to refactor
    }

    fn offset_values(&mut self, _index: &[usize], _offset: f64, _signs: &[i8]) {
        //no-op.  Will just use KKT matrix as passed to refactor
    }

    fn solve(&mut self, kkt: &CscMatrix<f64>, x: &mut [f64], b: &mut [f64]) {
        let (nzvals, rowvals, colptrs) = self.pardiso_kkt(kkt, &self.index32);
        let ps = &mut self.ps;
        ps.set_phase(Phase::SolveIterativeRefine);
        let n = b.len() as i32;

        assert_pardiso_const_rhs_config(ps);

        ps.pardiso(nzvals, colptrs, rowvals, b, x, n, 1_i32)
            .unwrap();
    }

    fn refactor(&mut self, kkt: &CscMatrix<f64>) -> bool {
        // Pardiso is quite robust and will usually produce some
        // kind of factorization unless there is an explicit
        // zero pivot or some other nastiness.   "success"
        // here just means that it didn't fail outright, although
        // the factorization could still be garbage.
        // PJG: would be preferable to propogate Errors here

        let (nzvals, rowvals, colptrs) = self.pardiso_kkt(kkt, &self.index32);
        let ps = &mut self.ps;
        ps.set_phase(Phase::NumFact);

        assert_pardiso_const_rhs_config(ps);

        ps.pardiso(
            nzvals,
            colptrs,
            rowvals,
            &mut [],
            &mut [],
            kkt.n as i32,
            1_i32,
        )
        .is_ok()
    }
}

fn assert_pardiso_const_rhs_config<P>(ps: &P)
where
    P: PardisoInterface,
{
    // pardiso wants b to be mutable since there is an option
    // to store the solution on b instead of x. We always want
    // to ensure that b is constant when calling pardiso
    assert!(
        ps.get_iparm(5) == 0,
        "Pardiso should be set to store its solution in x, not b [iparm[5] != 0 error]"
    );
}
