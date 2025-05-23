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

pub(crate) struct PardisoDirectLDLSolver<P>
where
    P: PardisoInterface + PardisoCustomInitialize,
{
    ps: P,
    nnzA: usize,
    nvars: usize,

    // Panua wants 32 bit CSC indices, 1 indexed
    // implemented as Option since I want to keep
    // the door open for a 64 bit 0-indexed version
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

// this trait is implement on the MKL and and Panua wrappers to
// allow for setting of different base defaults after pardisoinit
// initialization but before the user settings are applied
pub(crate) trait PardisoCustomInitialize {
    fn custom_iparm_initialize(&mut self, settings: &CoreSettings<f64>);
}

#[cfg(feature = "pardiso-panua")]
impl PardisoCustomInitialize for PanuaPardisoSolver {
    fn custom_iparm_initialize(&mut self, settings: &CoreSettings<f64>) {
        // disable internal iterative refinement if user enabled
        // iterative refinement is enabled in the settings.   It is
        // seemingly not possible to disable this completely within
        // MKL, and setting -99 there would mean "execute 99 high
        // accuracy refinements steps".   Not good.
        if settings.iterative_refinement_enable {
            self.set_iparm(7, -99); //# NB: 0 indexed
        }
        // request non-zeros in the factorization
        self.set_iparm(17, -1);
    }
}

#[cfg(feature = "pardiso-mkl")]
impl PardisoCustomInitialize for MKLPardisoSolver {
    fn custom_iparm_initialize(&mut self, _settings: &CoreSettings<f64>) {
        // request non-zeros in the factorization
        self.set_iparm(17, -1);
    }
}

impl<P> PardisoDirectLDLSolverInterface for PardisoDirectLDLSolver<P>
where
    P: PardisoInterface + PardisoCustomInitialize,
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
        settings: &CoreSettings<f64>,
        _perm: Option<Vec<usize>>,
    ) {
        // NB: ignore Dsigns here because pardiso doesn't
        // use information about the expected signs

        let (nzvals, rowvals, colptrs) = self.pardiso_kkt(kkt, &self.index32);
        let ps = &mut self.ps;

        // matrix is quasidefinite
        ps.set_matrix_type(MatrixType::RealSymmetricIndefinite);

        //init here gets the defaults
        ps.pardisoinit().unwrap();

        // overlay custom iparm initializations that might
        // be specific to MKL or Panua
        ps.custom_iparm_initialize(settings);

        // now apply user defined iparm settings if they exist.
        // Check here first for failed solves, because misuse of
        // this setting would likely be a disaster.
        for (i, &iparm) in settings.pardiso_iparm.iter().enumerate() {
            if iparm != i32::MIN {
                ps.set_iparm(i, iparm);
            }
        }

        if settings.pardiso_verbose {
            ps.set_message_level(MessageLevel::On);
        } else {
            ps.set_message_level(MessageLevel::Off);
        }

        //perform logical factorization
        ps.set_phase(Phase::Analysis);

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
        let nvars = KKT.n;
        let index32 = Some(PardisoMatrixIndices32::new(KKT));

        let mut solver = Self {
            ps,
            nnzA,
            nvars,
            index32,
        };

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

        let ps = PanuaPardisoSolver::new().unwrap();
        let nnzA = KKT.nnz();
        let nvars = KKT.n;
        let index32 = Some(PardisoMatrixIndices32::new(KKT));

        let mut solver = Self {
            ps,
            nnzA,
            nvars,
            index32,
        };

        solver.initialize(KKT, Dsigns, settings, perm);

        // Note : Panua doesn't support setting the number of threads
        // Always reads instead from ENV["OMP_NUM_THREADS"] before loading

        solver
    }
}

impl<P> DirectLDLSolverReqs for PardisoDirectLDLSolver<P>
where
    P: PardisoInterface + PardisoCustomInitialize,
{
    fn required_matrix_shape() -> MatrixTriangle {
        MatrixTriangle::Tril
    }
}

impl<P> HasLinearSolverInfo for PardisoDirectLDLSolver<P>
where
    P: PardisoInterface + PardisoCustomInitialize,
{
    fn linear_solver_info(&self) -> LinearSolverInfo {
        LinearSolverInfo {
            name: self.ps.name().to_string(),
            threads: self.ps.get_num_threads().unwrap() as usize,
            direct: true,
            nnzA: self.nnzA,
            nnzL: self.ps.get_iparm(17) as usize - self.nvars,
        }
    }
}

impl<P> DirectLDLSolver<f64> for PardisoDirectLDLSolver<P>
where
    P: PardisoInterface + PardisoCustomInitialize,
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

// returns true if the pardiso iparm settings are configured
// in a way that makes sense for clarabel
pub(crate) fn pardiso_iparm_is_valid(_iparm: &[i32; 64]) -> bool
where
{
    // placeholder for possible pardiso iparm checks.
    // NB: call comes from user settings, so need
    // to all u32:MIN values to be treated as "unset"
    true
}
