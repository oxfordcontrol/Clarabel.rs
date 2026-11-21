#![allow(non_snake_case)]
#![cfg(all(feature = "sdp", feature = "mpfr"))]
//! 50-digit SDP regression for the MPFR backend.
//!
//! Problem: minimise ½‖x‖² subject to b − x ∈ S₊ⁿ (svec, upper triangle,
//! column order, off-diagonals scaled by √2). Since svec is an isometry,
//! the optimum is x* = b − svec(Π_{S₊}(mat b)), the projection onto the
//! PSD cone. The reference values below were computed independently with
//! mpmath (`eigsy`) at 70 decimal digits; the solver must agree to 1e-50.
//!
//! n = 3 exercises the fixed-size 3×3 SVD / eigen kernels; n = 4 and 5
//! exercise the generic `xgesvd` / `xsyevr` paths of `native_lapack`.

use clarabel::{algebra::*, solver::*};
use num_traits::{Num, Signed, Zero};

const PREC_BITS: u32 = 256; // ≈ 77 decimal digits working precision

fn mp(s: &str) -> MpfrFloat {
    MpfrFloat::from_str_radix(s, 10).unwrap()
}

fn tri(n: usize) -> usize {
    n * (n + 1) / 2
}

fn identity(m: usize) -> CscMatrix<MpfrFloat> {
    CscMatrix::new(
        m,
        m,
        (0..=m).collect(),
        (0..m).collect(),
        (0..m).map(|_| MpfrFloat::from(1.0)).collect(),
    )
}

fn solve_projection(n: usize, b: &[i64]) -> (Vec<MpfrFloat>, MpfrFloat, SolverStatus) {
    let m = tri(n);
    assert_eq!(b.len(), m);
    let P = identity(m);
    let A = identity(m);
    let c = vec![MpfrFloat::zero(); m];
    let b: Vec<MpfrFloat> = b.iter().map(|&v| MpfrFloat::from(v)).collect();
    let cones = vec![SupportedConeT::PSDTriangleConeT(n)];

    let tol = mp("1e-60");
    let settings = DefaultSettingsBuilder::<MpfrFloat>::default()
        .max_iter(400)
        .tol_feas(tol.clone())
        .tol_gap_abs(tol.clone())
        .tol_gap_rel(tol.clone())
        .tol_ktratio(mp("1e-50"))
        .static_regularization_constant(mp("1e-70"))
        .static_regularization_proportional(mp("1e-140"))
        .dynamic_regularization_eps(mp("1e-120"))
        .dynamic_regularization_delta(mp("1e-70"))
        .iterative_refinement_reltol(mp("1e-75"))
        .iterative_refinement_abstol(mp("1e-75"))
        .iterative_refinement_max_iter(40)
        .verbose(false)
        .build()
        .unwrap();

    let mut solver = DefaultSolver::new(&P, &c, &A, &b, &cones, settings).unwrap();
    solver.solve();
    (
        solver.solution.x.clone(),
        solver.solution.obj_val.clone(),
        solver.solution.status,
    )
}

fn check(n: usize, b: &[i64], xref: &[&str], objref: &str) {
    set_mpfr_default_precision(PREC_BITS);
    let (x, obj, status) = solve_projection(n, b);
    assert_eq!(status, SolverStatus::Solved);
    let err = x
        .iter()
        .zip(xref)
        .map(|(xi, r)| (xi.clone() - mp(r)).abs())
        .fold(MpfrFloat::zero(), |a, e| if a > e { a } else { e });
    let objerr = (obj.clone() - mp(objref)).abs();
    println!("n = {n}: max |x - xref| = {err:e}, |obj - objref| = {objerr:e}");
    assert!(err <= mp("1e-50"), "x error {err:e} exceeds 1e-50");
    assert!(objerr <= mp("1e-50"), "obj error {objerr:e} exceeds 1e-50");
}

#[test]
fn sdp_projection_50_digits_n3() {
    check(
        3,
        &[-3, 1, 4, 1, 2, 5],
        &[
            "-3.07298332662265200773383793925147701065579607547831242155169",
            "0.369600416718895884902159066772597446495745273666830764536143",
            "-0.0222266855233666360650272173323101114292537279616942160279567",
            "0.314412131284437303321265491811422929650691233302999846240322",
            "-0.0267397008952522391203281479151516874528227072602297569720389",
            "-0.0160845305346105395635828049701450069762398497205751933991739",
        ],
        "4.84007686565713274555312959316409127630498257921732776518988",
    );
}

#[test]
fn sdp_projection_50_digits_n4() {
    check(
        4,
        &[-3, 1, 4, 1, 2, 5, -9, 2, 6, 5],
        &[
            "-4.92161460897880209907812285754718233751262888776084598129454",
            "0.755551972279106641432195962041972572306087139381945327723301",
            "-0.0579950715536925006620515684423185938501125266539690154266583",
            "1.82075557585291733446254316768299407582046927860334180896837",
            "-0.197648440251756207774870609187016427191421937658885674988464",
            "-0.336795049022271717797703319110480271122686245250898803208363",
            "-4.34061106197930214000376301655951922065163456856670221618175",
            "0.471186257791820977188982465342604368226497204925120746126338",
            "1.13548112852097464508879532314725995326423455302615989902963",
            "-1.9140979016321668716044711102791126957116118477812477754183",
        ],
        "26.1400841498005819964324460409398415193063857396404828039123",
    );
}

#[test]
fn sdp_projection_50_digits_n5() {
    check(
        5,
        &[-3, 1, 4, 1, 2, 5, -9, 2, 6, 5, 3, 5, -8, 9, 7],
        &[
            "-4.55947380166770938405093070486978829300307642066839822880997",
            "-0.0159093272891452627855328354486298252960089050084411565697711",
            "-0.292079673822958561731536872386742127864213778647433236715208",
            "2.71400597325436675835395347475510137891483205059786147891076",
            "0.601038053979965015958654851885182286992581583032381694323494",
            "-1.4125090165112095527704359633804118681286734358439752171344",
            "-4.9117361671558630231261750561379514678608901658027670141353",
            "-0.416462821260488092152126457339907643502635644491426638809636",
            "2.64921168810381908076789187822175582132736153678388472628354",
            "-2.92551302013882902596047801694731468027224727372691943277481",
            "3.35518962882065303260932306859654948400548436088615142821472",
            "0.612927784814257134510154032373555880081724511782696764179785",
            "-2.28229779423326868559622414074867136816064417804175306489403",
            "3.14771641232776550002704511833877076679213097601558206327205",
            "-1.86041314296793157659589300589766529359165604071709057123744",
        ],
        "50.3416405727323603833585832219024806242782409500935488018962",
    );
}
