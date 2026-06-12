#![allow(non_snake_case)]
#![allow(clippy::type_complexity)]
#![cfg(all(feature = "sdp", feature = "mpfr"))]
use clarabel::{algebra::*, solver::*};
use num_traits::{Zero, Signed};

fn basic_sdp_data() -> (
    CscMatrix<MpfrFloat>,
    Vec<MpfrFloat>,
    CscMatrix<MpfrFloat>,
    Vec<MpfrFloat>,
    Vec<SupportedConeT<MpfrFloat>>,
) {
    let P = CscMatrix::<f64>::identity(6).to_mpfr();
    let A = CscMatrix::<f64>::identity(6).to_mpfr();
    let c = vec![MpfrFloat::zero(); 6];
    let b = vec![-3., 1., 4., 1., 2., 5.].into_iter().map(MpfrFloat::from).collect();
    let cones = vec![SupportedConeT::PSDTriangleConeT(3)];
    (P, c, A, b, cones)
}

fn basic_sdp_solution() -> (Vec<MpfrFloat>, MpfrFloat) {
    let refsol = vec![
        -3.0729833267361095,
        0.3696004167288786,
        -0.022226685581313674,
        0.31441213129613066,
        -0.026739700851545107,
        -0.016084530571308823,
    ].into_iter().map(MpfrFloat::from).collect();
    let refobj = MpfrFloat::from(4.840076866013861);
    (refsol, refobj)
}

trait ToMpfr { fn to_mpfr(&self) -> CscMatrix<MpfrFloat>; }
impl ToMpfr for CscMatrix<f64> {
    fn to_mpfr(&self) -> CscMatrix<MpfrFloat> {
        let nzval = self.nzval.iter().map(|&v| MpfrFloat::from(v)).collect();
        CscMatrix::new(self.m, self.n, self.colptr.clone(), self.rowval.clone(), nzval)
    }
}

#[test]
fn test_sdp_feasible_mpfr() {
    let (P, c, A, b, cones) = basic_sdp_data();
    let (_refsol, _refobj) = basic_sdp_solution();
    let settings = DefaultSettingsBuilder::default().build().unwrap();
    let mut solver = DefaultSolver::new(&P, &c, &A, &b, &cones, settings).unwrap();
    solver.solve();
    assert_eq!(solver.solution.status, SolverStatus::Solved);
    let dist = solver.solution.x.iter().zip(&_refsol).map(|(x, r)| (x.clone() - r.clone()).abs()).fold(MpfrFloat::zero(), |a, b| if a > b { a } else { b });
    assert!(dist <= MpfrFloat::from(1e-5));
}
