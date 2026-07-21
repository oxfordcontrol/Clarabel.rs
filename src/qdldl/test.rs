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

fn inf_norm<T: FloatT>(a: &[T]) -> T {
    a.iter().fold(T::zero(), |acc, x| T::max(acc, T::abs(*x)))
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
fn test_solve_refined() {
    let A = test_matrix_4x4();
    let mut factors = QDLDLFactorisation::new(&A, None).unwrap();
    let x_true = [1., -2., 3., -4.];
    let b = [20.0, -22.0, 32.0, -7.0];

    // agrees with the plain solve
    let mut x = [0.0; 4];
    assert!(factors.solve_refined(&mut x, &b, 1e-13, 1e-12, 10, 5.0));
    assert!(inf_norm_diff(&x_true, &x) <= 1e-10);

    // refines against the values currently in the internal workspace,
    // even where they differ from the factored ones: scale the matrix
    // by 1.1 without refactoring, and refinement (contraction rate
    // ||I - A⁻¹(1.1A)|| = 0.1 per pass) must converge to the solution
    // of the *scaled* system using the stale factors
    let indices: Vec<usize> = (0..A.nzval.len()).collect();
    factors.scale_values(&indices, 1.1);
    let mut x = [0.0; 4];
    assert!(factors.solve_refined(&mut x, &b, 1e-13, 1e-12, 20, 5.0));
    let x_scaled: Vec<f64> = x_true.iter().map(|v| v / 1.1).collect();
    assert!(inf_norm_diff(&x_scaled, &x) <= 1e-10);
}

// A small quasidefinite KKT-like matrix [[diag(p) B'; B -diag(r)]] in
// upper triangular CSC form, with the D signs it should factor with.
#[cfg(test)]
fn test_matrix_kkt_like(nx: usize, nz: usize, seed: u64) -> (CscMatrix<f64>, Vec<i8>) {
    let mut state = seed;
    let mut next = move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 33) as f64) / ((1u64 << 31) as f64) - 1.0
    };

    let n = nx + nz;
    let mut cols: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
    for j in 0..nx {
        cols[j].push((j, 1.0 + next().abs()));
    }
    for j in 0..nz {
        let col = nx + j;
        for t in 0..3 {
            let i = ((next().abs() * nx as f64) as usize + t * 7) % nx;
            cols[col].push((i, next()));
        }
        cols[col].sort_by_key(|e| e.0);
        cols[col].dedup_by_key(|e| e.0);
        cols[col].push((col, -(1.0 + next().abs())));
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
    let mut signs = vec![1i8; n];
    signs[nx..].fill(-1);
    (
        CscMatrix {
            m: n,
            n,
            colptr,
            rowval,
            nzval,
        },
        signs,
    )
}

// With exactly-representable values every product and sum below is
// exact, so the both-triangles residual must agree with the triangular
// symv *bitwise*.  Any disagreement would be a structural error in the
// copy or its value map rather than a rounding difference.
#[test]
fn test_symmetric_copy_exact_on_integer_data() {
    // upper triangular, small integers, some zero-free columns
    //   [ 2  -1   0   3 ]
    //   [    -4   5   0 ]
    //   [         6  -2 ]
    //   [             8 ]
    let triu = CscMatrix {
        m: 4,
        n: 4,
        colptr: vec![0, 1, 3, 5, 8],
        rowval: vec![0, 0, 1, 1, 2, 0, 2, 3],
        nzval: vec![2., -1., -4., 5., 6., 3., -2., 8.],
    };

    let mut sym = _build_symmetric_copy(&triu).unwrap();
    for (dst, &k) in zip(&mut sym.A.nzval, &sym.sym_to_triu) {
        *dst = triu.nzval[k as usize];
    }
    // 4 diagonal entries, 4 off-diagonal mirrored
    assert_eq!(sym.A.nzval.len(), 2 * 8 - 4);

    let x = [1., -2., 4., 8.];
    let b = [100., -100., 50., 25.];

    let mut e_ref = b;
    triu.sym_up().symv(&mut e_ref, &x, -1.0, 1.0);

    let mut e_new = [0.0; 4];
    let norme = _sym_residual(&sym, &mut e_new, &b, &x);

    assert_eq!(e_ref, e_new); // bitwise on exact data
    assert_eq!(norme, inf_norm(&e_new));
}

// A non-finite solution must be reported rather than masked.  The running
// maximum cannot see a NaN on its own, because IEEE maxNum returns the
// non-NaN operand, so without explicit tracking a NaN residual would leave
// the norm finite and the caller would accept a NaN solution as converged.
#[test]
fn test_sym_residual_reports_nan() {
    let triu = CscMatrix {
        m: 3,
        n: 3,
        colptr: vec![0, 1, 3, 6],
        rowval: vec![0, 0, 1, 0, 1, 2],
        nzval: vec![2.0, 1.0, 3.0, -1.0, 0.5, 4.0],
    };
    let mut sym = _build_symmetric_copy(&triu).unwrap();
    for (dst, &k) in zip(&mut sym.A.nzval, &sym.sym_to_triu) {
        *dst = triu.nzval[k as usize];
    }
    let b = [1.0, 2.0, 3.0];
    let mut e = [0.0; 3];

    // finite input: ordinary infinity norm
    let n_ok: f64 = _sym_residual(&sym, &mut e, &b, &[1.0, 1.0, 1.0]);
    assert!(n_ok.is_finite());

    // one NaN in x taints its rows, and must be reported
    let norme: f64 = _sym_residual(&sym, &mut e, &b, &[1.0, f64::NAN, 1.0]);
    assert!(norme.is_nan(), "NaN residual must not be masked");
}

// The both-triangles copy must reproduce the matrix the triangular
// symv represents, so that refinement residuals are unchanged in value
// (they differ only in summation order).
#[test]
fn test_symmetric_copy_matches_triangular_symv() {
    let (A, signs) = test_matrix_kkt_like(50, 35, 777);
    let opts = QDLDLSettingsBuilder::<f64>::default()
        .Dsigns(signs)
        .build()
        .unwrap();
    let factors = QDLDLFactorisation::new(&A, Some(opts)).unwrap();

    // the permuted internal matrix, and its both-triangles copy
    let triu = &factors.workspace.triuA;
    let mut sym = _build_symmetric_copy(triu).unwrap();
    for (dst, &k) in zip(&mut sym.A.nzval, &sym.sym_to_triu) {
        *dst = triu.nzval[k as usize];
    }

    let n = triu.ncols();
    // every off-diagonal entry must appear on both sides
    assert_eq!(sym.A.nzval.len(), 2 * triu.nzval.len() - n);

    let x: Vec<f64> = (0..n).map(|i| ((i * 13) % 7) as f64 - 3.0).collect();
    let b: Vec<f64> = (0..n).map(|i| ((i * 5) % 11) as f64 - 5.0).collect();

    // reference: e = b - A*x via the triangular symv
    let mut e_ref = b.clone();
    triu.sym_up().symv(&mut e_ref, &x, -1.0, 1.0);

    let mut e_new = vec![0.0; n];
    let norme = _sym_residual(&sym, &mut e_new, &b, &x);

    // same values up to summation order, and the returned norm agrees
    assert!(inf_norm_diff(&e_ref, &e_new) <= 1e-10 * inf_norm(&e_ref).max(1.0));
    assert!((norme - inf_norm(&e_new)).abs() <= 1e-12 * norme.max(1.0));
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
