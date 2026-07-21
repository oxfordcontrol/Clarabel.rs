use super::*;
extern crate amd;

#[cfg(test)]
fn test_matrix_4x4() -> CscMatrix<f64> {
    // A =
    //[ 8.0  -3.0   2.0    ⋅ ]
    //[  ⋅    8.0  -1.0    ⋅ ]
    //[  ⋅     ⋅    8.0  -1.0]
    //[  ⋅     ⋅     ⋅    1.0]
    let Ap = vec![0, 1, 3, 6, 8];
    let Ai = vec![0, 0, 1, 0, 1, 2, 2, 3];
    let Ax = vec![8., -3., 8., 2., -1., 8., -1., 1.];
    CscMatrix {
        m: 4,
        n: 4,
        colptr: Ap,
        rowval: Ai,
        nzval: Ax,
    }
}

fn inf_norm_diff<T: FloatT>(a: &[T], b: &[T]) -> T {
    zip(a, b).fold(T::zero(), |acc, (x, y)| T::max(acc, T::abs(*x - *y)))
}

// tests some of the private functions of QDLDL.  Configured
// as submodule from lib.rs to expose internals.

#[test]
fn test_invperm() {
    let perm = vec![3, 0, 2, 1];
    assert!(_invperm(&perm).is_ok())
}

//test fail on bad permutation
#[test]
fn test_invperm_bad_perm1() {
    let perm = vec![3, 0, 2, 0]; //repeated index
    assert!(_invperm(&perm).is_err())
}

#[test]
fn test_invperm_bad_perm2() {
    let perm = vec![4, 0, 2, 1]; //index too big
    assert!(_invperm(&perm).is_err())
}

#[test]
fn test_permute() {
    let perm = vec![3, 0, 2, 1];
    let b = vec![1., 2., 3., 4.];
    let mut x = vec![0.; 4];
    let mut y = vec![0.; 4];

    permute(&mut x, &b, &perm);
    assert_eq!(x, vec![4., 1., 3., 2.]);

    ipermute(&mut y, &x, &perm);
    assert_eq!(y, b);
}

#[test]
fn test_solve_from_factors() {
    //L =
    //[ ⋅    ⋅     ⋅    ⋅ ]
    //[1.0   ⋅     ⋅    ⋅ ]
    //[2.0  1.0    ⋅    ⋅ ]
    //[ ⋅   7.0  -3.0   ⋅ ]

    let Lp = [0, 2, 4, 5, 5];
    let Li = [1, 2, 2, 3, 3];
    let Lx = [1., 2., 1., 7., -3.];
    let _d = [4., -1., -2., 1.];
    let dinv = [0.25, -1.0, -0.5, 1.0];
    let x = [-3., 2., 1., 4.];

    //(I+L)x = b.  Back solve on b in place.
    let mut b = [-3., -1., -3., 15.];
    _lsolve_unsafe(&Lp, &Li, &Lx, &mut b);
    assert_eq!(b, x);

    let mut b = [-3., -1., -3., 15.];
    _lsolve_safe(&Lp, &Li, &Lx, &mut b);
    assert_eq!(b, x);

    //(I+L')x = b.  Back solve on b in place.
    let mut b = [1., 31., -11., 4.];
    _ltsolve_unsafe(&Lp, &Li, &Lx, &mut b);
    assert_eq!(b, x);

    let mut b = [1., 31., -11., 4.];
    _ltsolve_safe(&Lp, &Li, &Lx, &mut b);
    assert_eq!(b, x);

    //(I+L)*D*(I+L)*x = b.  Back solve on b in place;
    let mut b = vec![4., -27., -1., -279.];
    _solve(&Lp, &Li, &Lx, &dinv, &mut b);
    assert_eq!(b, x);
}

#[test]
fn test_etree() {
    let n = 4;
    let A = test_matrix_4x4();
    let mut Lnz = vec![0; n];
    let mut iwork = vec![0; 3 * n];
    let mut etree = vec![0; n];

    let _out = _etree(
        A.nrows(),
        &A.colptr,
        &A.rowval,
        &mut iwork,
        &mut Lnz,
        &mut etree,
    )
    .unwrap();

    assert_eq!(etree, vec![1, 2, 3, QDLDL_UNKNOWN]);
}

#[test]
fn test_amd() {
    let A = test_matrix_4x4();
    let (perm, iperm, _) = get_amd_ordering(&A, 1.5);
    assert_eq!(perm, [3, 0, 1, 2]);
    assert_eq!(iperm, [1, 2, 3, 0]);
}

#[test]
fn test_permute_symmetric() {
    //no permutation at all
    let A = test_matrix_4x4();
    let iperm: Vec<usize> = vec![0, 1, 2, 3];
    let (P, AtoPAPt) = permute_symmetric(&A, &iperm);

    assert_eq!(&A.colptr, &P.colptr);
    assert_eq!(&A.rowval, &P.rowval);
    assert_eq!(&A.nzval, &P.nzval);
    let linearidx: Vec<usize> = (0..AtoPAPt.len()).collect();
    assert_eq!(&linearidx, &AtoPAPt);

    //test with a permutation.  NB: the permutation
    //implemented in QDLDL produces a result in which entries
    //are not ordering by increasing row number within
    //each column, so caution is required when comparing
    //w.r.t. other tools (i.e. Matlab/Julia/Python etc)

    let mut A = test_matrix_4x4();

    //set the problem data to increasing values columnwise
    for i in 0..A.nzval.len() {
        A.nzval[i] = i as f64 + 1.;
    }

    let perm: Vec<usize> = vec![2, 3, 0, 1];
    let iperm = _invperm(&perm).unwrap();
    let (P, _) = permute_symmetric(&A, &iperm);

    assert_eq!(&P.colptr, &vec![0, 1, 3, 5, 8]);
    assert_eq!(&P.rowval, &vec![0, 0, 1, 2, 0, 2, 3, 0]);
    assert_eq!(&P.nzval, &vec![6.0, 7.0, 8.0, 1.0, 4.0, 2.0, 3.0, 5.0]);
}

#[test]
fn test_settings_builder() {
    //NB: the default regularize_eps is 1e-12.  Use this
    //as a reference point throughout
    let expected_regularize_eps = 1e-12;

    //check that defaults appear when not using builder
    let opts = QDLDLSettings::<f64>::default();
    assert_eq!(opts.regularize_eps, expected_regularize_eps);

    //same thing through the builder
    let opts = QDLDLSettingsBuilder::<f64>::default().build().unwrap();
    assert_eq!(opts.regularize_eps, expected_regularize_eps);

    //and now a custom builder
    let opts = QDLDLSettingsBuilder::default()
        .perm(vec![0, 1, 2, 3])
        .logical(false)
        .regularize_enable(true)
        .regularize_eps(1e-3)
        .regularize_delta(1e-3)
        .build()
        .unwrap();

    assert_eq!(opts.regularize_eps, 1e-3);
    assert_eq!(opts.regularize_delta, 1e-3);
}

#[test]
fn test_solve_basic() {
    let A = test_matrix_4x4();

    //default settings but no permutation
    let opts = QDLDLSettingsBuilder::default()
        .perm(vec![0, 1, 2, 3])
        .build()
        .unwrap();

    let mut factors = QDLDLFactorisation::new(&A, Some(opts)).unwrap();
    let x = [1., -2., 3., -4.];
    let mut b = [20.0, -22.0, 32.0, -7.0];
    //solves in place
    factors.solve(&mut b);
    assert!(inf_norm_diff(&x, &b) <= 1e-8);

    //now with all defaults, including amd
    let mut factors = QDLDLFactorisation::new(&A, None).unwrap();
    let x = [1., -2., 3., -4.];
    let mut b = [20.0, -22.0, 32.0, -7.0];
    //solves in place
    factors.solve(&mut b);
    assert!(inf_norm_diff(&x, &b) <= 1e-8);

    //user specified permutation
    let opts = QDLDLSettingsBuilder::<f64>::default()
        .perm(vec![3, 0, 2, 1])
        .build()
        .unwrap();
    let mut factors = QDLDLFactorisation::new(&A, Some(opts)).unwrap();
    let x = [1., -2., 3., -4.];
    let mut b = [20.0, -22.0, 32.0, -7.0];
    //solves in place
    factors.solve(&mut b);
    assert!(inf_norm_diff(&x, &b) <= 1e-8);
}

#[test]
#[should_panic]
fn test_solve_logical() {
    let A = test_matrix_4x4();
    //logical first, then refactor and solve
    let opts = QDLDLSettingsBuilder::default()
        .logical(true)
        .build()
        .unwrap();

    let mut factors = QDLDLFactorisation::new(&A, Some(opts)).unwrap();
    let mut b = [20.0, -22.0, 32.0, -7.0];
    //solves in place
    factors.solve(&mut b); //should panic
}

#[test]
fn test_solve_logical_refactor() {
    let A = test_matrix_4x4();
    //logical first, then refactor and solve
    let opts = QDLDLSettingsBuilder::default()
        .logical(true)
        .build()
        .unwrap();

    let mut factors = QDLDLFactorisation::new(&A, Some(opts)).unwrap();
    let x = [1., -2., 3., -4.];
    let mut b = [20.0, -22.0, 32.0, -7.0];
    //solves in place
    assert!(factors.refactor().is_ok());
    factors.solve(&mut b);
    assert!(inf_norm_diff(&x, &b) <= 1e-8);
}

// Build a deterministic pseudo-random quasidefinite KKT-like matrix
// [[diag(p) B'; B -diag(r)]] in upper triangular CSC form, sized so that
// its factor has genuine fill-in.  Returns (A, Dsigns).
#[cfg(test)]
fn test_matrix_quasidef(nx: usize, nz: usize, seed: u64) -> (CscMatrix<f64>, Vec<i8>) {
    // simple LCG so the test needs no rand dependency
    let mut state = seed;
    let mut next = move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 33) as f64) / ((1u64 << 31) as f64) - 1.0 // in [-1,1)
    };

    let n = nx + nz;
    let mut cols: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
    for j in 0..nx {
        cols[j].push((j, 1.0 + next().abs())); // positive definite block
    }
    for j in 0..nz {
        let col = nx + j;
        // a few entries of B in each column, rows in 0..nx
        for t in 0..3 {
            let i = ((next().abs() * nx as f64) as usize + t * 7) % nx;
            cols[col].push((i, next()));
        }
        cols[col].sort_by_key(|e| e.0);
        cols[col].dedup_by_key(|e| e.0);
        cols[col].push((col, -(1.0 + next().abs()))); // negative definite block
    }

    let mut colptr = vec![0usize];
    let (mut rowval, mut nzval) = (Vec::new(), Vec::new());
    for c in &cols {
        for &(i, v) in c {
            rowval.push(i);
            nzval.push(v);
        }
        colptr.push(rowval.len());
    }
    let A = CscMatrix {
        m: n,
        n,
        colptr,
        rowval,
        nzval,
    };
    let mut signs = vec![1i8; n];
    signs[nx..].fill(-1);
    (A, signs)
}

// Refactorization must reproduce _factor_inner bit-for-bit: the scheduled
// replay path performs the identical operations in the identical order, so
// L, D, Dinv, the inertia and the regularization count of a refactor must
// all equal those of a fresh factorization of the same values.
#[test]
fn test_refactor_matches_fresh_factor_exactly() {
    let (A, signs) = test_matrix_quasidef(40, 30, 12345);

    let opts = || {
        QDLDLSettingsBuilder::<f64>::default()
            .Dsigns(signs.clone())
            .build()
            .unwrap()
    };

    let mut f1 = QDLDLFactorisation::new(&A, Some(opts())).unwrap();

    // change every value, refactor (first refactor builds the schedule
    // and replays it), and compare against a fresh factorization
    let mut A2 = A.clone();
    for v in A2.nzval.iter_mut() {
        *v *= 1.25;
    }
    let indices: Vec<usize> = (0..A2.nzval.len()).collect();

    f1.update_values(&indices, &A2.nzval);
    f1.refactor().unwrap();
    assert!(f1.refactor_schedule_is_ready()); // replay path, not a fallback

    let f2 = QDLDLFactorisation::new(&A2, Some(opts())).unwrap();

    assert_eq!(f1.perm, f2.perm); // same AMD ordering on the same pattern
    assert_eq!(f1.L.nzval, f2.L.nzval); // bitwise
    assert_eq!(f1.D, f2.D);
    assert_eq!(f1.Dinv, f2.Dinv);
    assert_eq!(f1.positive_inertia(), f2.positive_inertia());
    assert_eq!(f1.regularize_count(), f2.regularize_count());

    // and again, to exercise the replay path on an already-built schedule
    f1.update_values(&indices, &A.nzval);
    f1.refactor().unwrap();
    let f3 = QDLDLFactorisation::new(&A, Some(opts())).unwrap();
    assert_eq!(f1.L.nzval, f3.L.nzval);
    assert_eq!(f1.D, f3.D);
}

// Same bit-identity requirement when dynamic regularization fires: the
// pivot tests happen in the same order on the same values, so the same
// pivots must be perturbed.
#[test]
fn test_refactor_matches_fresh_factor_with_regularization() {
    let (mut A, signs) = test_matrix_quasidef(40, 30, 999);
    // shrink some diagonal entries so that regularization triggers
    for j in 0..40 {
        let d = A.colptr[j]; // diagonal of the (j,j) leading block column
        A.nzval[d] *= 1e-14;
    }

    let opts = || {
        QDLDLSettingsBuilder::<f64>::default()
            .Dsigns(signs.clone())
            .regularize_eps(1e-12)
            .regularize_delta(1e-7)
            .build()
            .unwrap()
    };

    let mut f1 = QDLDLFactorisation::new(&A, Some(opts())).unwrap();

    let mut A2 = A.clone();
    for v in A2.nzval.iter_mut() {
        *v *= 0.75;
    }
    let indices: Vec<usize> = (0..A2.nzval.len()).collect();
    f1.update_values(&indices, &A2.nzval);
    f1.refactor().unwrap();
    assert!(f1.refactor_schedule_is_ready()); // replay path, not a fallback

    let f2 = QDLDLFactorisation::new(&A2, Some(opts())).unwrap();
    assert!(f2.regularize_count() > 0); // the scenario is actually exercised
    assert_eq!(f1.L.nzval, f2.L.nzval);
    assert_eq!(f1.D, f2.D);
    assert_eq!(f1.Dinv, f2.Dinv);
    assert_eq!(f1.positive_inertia(), f2.positive_inertia());
    assert_eq!(f1.regularize_count(), f2.regularize_count());
}

// Both replay paths must produce the same factorization: the block-wise and
// entry-wise updates perform the same operations on the same values in the
// same order, and which one is selected is a performance decision only.  The
// two matrices below sit on opposite sides of that decision.
#[test]
fn test_both_replay_paths_agree_with_a_fresh_factor() {
    // banded, so the factor's rows are long consecutive blocks
    let n = 120usize;
    let (mut colptr, mut rowval, mut nzval) = (vec![0usize], Vec::new(), Vec::new());
    for j in 0..n {
        for i in j.saturating_sub(20)..=j {
            rowval.push(i);
            nzval.push(if i == j {
                40.0
            } else {
                -1.0 / (1 + j - i) as f64
            });
        }
        colptr.push(rowval.len());
    }
    let banded = CscMatrix {
        m: n,
        n,
        colptr,
        rowval,
        nzval,
    };

    // and a sparse quasidefinite matrix, whose blocks are short
    let (scattered, signs) = test_matrix_quasidef(60, 45, 4242);

    for (name, A, ds) in [
        ("banded", banded, None),
        ("scattered", scattered, Some(signs)),
    ] {
        let mut b = QDLDLSettingsBuilder::<f64>::default();
        if let Some(ds) = ds {
            b.Dsigns(ds);
        }
        let opts = b.build().unwrap();

        let mut f = QDLDLFactorisation::new(&A, Some(opts.clone())).unwrap();
        let mut scaled = A.clone();
        for v in scaled.nzval.iter_mut() {
            *v *= 1.25;
        }
        let idx: Vec<usize> = (0..A.nzval.len()).collect();
        f.update_values(&idx, &scaled.nzval);
        f.refactor().unwrap();
        assert!(
            f.refactor_schedule_is_ready(),
            "{name}: replay path not engaged"
        );

        let fresh = QDLDLFactorisation::new(&scaled, Some(opts)).unwrap();
        assert_eq!(f.L.nzval, fresh.L.nzval, "{name}: L differs");
        assert_eq!(f.D, fresh.D, "{name}: D differs");
        assert_eq!(f.Dinv, fresh.Dinv, "{name}: Dinv differs");
    }
}

// The block-wise path must be chosen for a factor whose updates are long
// consecutive blocks, and declined for one whose blocks are short -- which is
// what makes the two cases above cover both paths rather than one twice.
#[test]
fn test_run_selection_follows_block_structure() {
    let n = 120usize;
    let (mut colptr, mut rowval, mut nzval) = (vec![0usize], Vec::new(), Vec::new());
    for j in 0..n {
        for i in j.saturating_sub(20)..=j {
            rowval.push(i);
            nzval.push(if i == j {
                40.0
            } else {
                -1.0 / (1 + j - i) as f64
            });
        }
        colptr.push(rowval.len());
    }
    let banded = CscMatrix {
        m: n,
        n,
        colptr,
        rowval,
        nzval,
    };
    let mut f = QDLDLFactorisation::new(&banded, None).unwrap();
    f.refactor().unwrap();
    let banded_cols = f.columns_using_blocks();
    assert!(
        banded_cols > n / 2,
        "long blocks should select the block path for most columns, got {banded_cols}"
    );

    let (scattered, signs) = test_matrix_quasidef(60, 45, 4242);
    let opts = QDLDLSettingsBuilder::<f64>::default()
        .Dsigns(signs)
        .build()
        .unwrap();
    let mut f = QDLDLFactorisation::new(&scattered, Some(opts)).unwrap();
    f.refactor().unwrap();
    assert_eq!(
        f.columns_using_blocks(),
        0,
        "short blocks should decline the block path"
    );
}

#[test]
fn test_bad_numeric_pivot() {
    //Disable regularization to force an exact zero pivot
    let opts = QDLDLSettingsBuilder::default()
        .regularize_enable(false)
        .build()
        .unwrap();

    //set the first element of A to zero (top left)
    let mut A = test_matrix_4x4();
    A.nzval[0] = 0.;
    assert!(QDLDLFactorisation::new(&A, Some(opts.clone())).is_err());

    //set the final element of A to zero (top left)
    let mut A = test_matrix_4x4();
    *A.nzval.last_mut().unwrap() = 0.;
    assert!(QDLDLFactorisation::new(&A, Some(opts)).is_err());
}

#[test]
fn test_lower_triangular() {
    //create a matrix with a logical zero on the diagonal
    let opts = QDLDLSettingsBuilder::default()
        .logical(true)
        .build()
        .unwrap();

    let A = CscMatrix::from(&[
        //
        [1.0, 3.0, 5.0],
        [2.0, 3.0, 6.0],
        [1.0, 4.0, 7.0],
    ]);
    assert!(QDLDLFactorisation::new(&A, Some(opts)).is_err());
}

#[test]
fn test_zero_column_error() {
    //create a matrix with a logical zero on the diagonal
    let opts = QDLDLSettingsBuilder::default()
        .logical(true)
        .build()
        .unwrap();

    let A = CscMatrix::from(&[
        //
        [1.0, 0.0, 5.0],
        [0.0, 0.0, 6.0],
        [1.0, 0.0, 7.0],
    ]);

    assert!(QDLDLFactorisation::new(&A, Some(opts)).is_err());
}
