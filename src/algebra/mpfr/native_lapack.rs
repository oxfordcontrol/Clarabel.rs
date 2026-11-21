use super::MpfrFloat;
use crate::algebra::*;
use crate::algebra::dense::private;
use num_traits::{Zero, One, FromPrimitive, Signed};
use std::cmp::Ordering;

impl private::BlasFloatSealed for MpfrFloat {}
impl BlasFloatT for MpfrFloat {}

impl XpotrfScalar for MpfrFloat {
    fn xpotrf(uplo: u8, n: i32, a: &mut [Self], lda: i32, info: &mut i32) {
        let is_u = uplo == b'U' || uplo == b'u';
        *info = 0;
        for j in 0..n {
            let mut sum = MpfrFloat::zero();
            for k in 0..j {
                let akj = if is_u { a[(j * lda + k) as usize].clone() } else { a[(k * lda + j) as usize].clone() };
                sum = sum + akj.clone() * akj;
            }
            let ajj_idx = (j * lda + j) as usize;
            let diag = a[ajj_idx].clone() - sum;
            if diag <= MpfrFloat::zero() {
                *info = j + 1;
                return;
            }
            let diag_sqrt = diag.sqrt();
            a[ajj_idx] = diag_sqrt.clone();
            for i in (j + 1)..n {
                let mut sum_ij = MpfrFloat::zero();
                for k in 0..j {
                    let aki = if is_u { a[(i * lda + k) as usize].clone() } else { a[(k * lda + i) as usize].clone() };
                    let akj = if is_u { a[(j * lda + k) as usize].clone() } else { a[(k * lda + j) as usize].clone() };
                    sum_ij = sum_ij + aki * akj;
                }
                let aij_idx = if is_u { (i * lda + j) as usize } else { (j * lda + i) as usize };
                a[aij_idx] = (a[aij_idx].clone() - sum_ij) / diag_sqrt.clone();
            }
        }
    }
}

impl XpotrsScalar for MpfrFloat {
    fn xpotrs(uplo: u8, n: i32, nrhs: i32, a: &[Self], lda: i32, b: &mut [Self], ldb: i32, info: &mut i32) {
        let is_u = uplo == b'U' || uplo == b'u';
        *info = 0;
        for rhs in 0..nrhs {
            let b_col = rhs * ldb;
            if is_u {
                for i in 0..n {
                    let mut sum = MpfrFloat::zero();
                    for j in 0..i { sum = sum + a[(i * lda + j) as usize].clone() * b[(b_col + j) as usize].clone(); }
                    b[(b_col + i) as usize] = (b[(b_col + i) as usize].clone() - sum) / a[(i * lda + i) as usize].clone();
                }
            } else {
                for i in 0..n {
                    let mut sum = MpfrFloat::zero();
                    for j in 0..i { sum = sum + a[(j * lda + i) as usize].clone() * b[(b_col + j) as usize].clone(); }
                    b[(b_col + i) as usize] = (b[(b_col + i) as usize].clone() - sum) / a[(i * lda + i) as usize].clone();
                }
            }
            if is_u {
                for i in (0..n).rev() {
                    let mut sum = MpfrFloat::zero();
                    for j in (i + 1)..n { sum = sum + a[(j * lda + i) as usize].clone() * b[(b_col + j) as usize].clone(); }
                    b[(b_col + i) as usize] = (b[(b_col + i) as usize].clone() - sum) / a[(i * lda + i) as usize].clone();
                }
            } else {
                for i in (0..n).rev() {
                    let mut sum = MpfrFloat::zero();
                    for j in (i + 1)..n { sum = sum + a[(i * lda + j) as usize].clone() * b[(b_col + j) as usize].clone(); }
                    b[(b_col + i) as usize] = (b[(b_col + i) as usize].clone() - sum) / a[(i * lda + i) as usize].clone();
                }
            }
        }
    }
}

impl XgesvScalar for MpfrFloat {
    fn xgesv(n: i32, nrhs: i32, a: &mut [Self], lda: i32, ipiv: &mut [i32], b: &mut [Self], ldb: i32, info: &mut i32) {
        *info = 0;
        for i in 0..n { ipiv[i as usize] = i + 1; }
        for k in 0..n {
            let mut max_val = MpfrFloat::zero();
            let mut max_idx = k;
            for i in k..n {
                let val = a[(k * lda + i) as usize].clone().abs();
                if val > max_val { max_val = val; max_idx = i; }
            }
            if max_val == MpfrFloat::zero() { *info = k + 1; return; }
            if max_idx != k {
                ipiv[k as usize] = max_idx + 1;
                for j in 0..n { a.swap((j * lda + k) as usize, (j * lda + max_idx) as usize); }
            }
            let akk = a[(k * lda + k) as usize].clone();
            for i in (k + 1)..n { a[(k * lda + i) as usize] = a[(k * lda + i) as usize].clone() / akk.clone(); }
            for j in (k + 1)..n {
                for i in (k + 1)..n {
                    let a_ik = a[(k * lda + i) as usize].clone();
                    let a_kj = a[(j * lda + k) as usize].clone();
                    a[(j * lda + i) as usize] = a[(j * lda + i) as usize].clone() - a_ik * a_kj;
                }
            }
        }
        for rhs in 0..nrhs {
            let b_col = rhs * ldb;
            for i in 0..n {
                let piv = (ipiv[i as usize] - 1) as usize;
                if piv != i as usize { b.swap((b_col + i) as usize, b_col as usize + piv); }
            }
            for i in 0..n {
                let mut sum = MpfrFloat::zero();
                for j in 0..i { sum = sum + a[(j * lda + i) as usize].clone() * b[(b_col + j) as usize].clone(); }
                b[(b_col + i) as usize] = b[(b_col + i) as usize].clone() - sum;
            }
            for i in (0..n).rev() {
                let mut sum = MpfrFloat::zero();
                for j in (i + 1)..n { sum = sum + a[(j * lda + i) as usize].clone() * b[(b_col + j) as usize].clone(); }
                b[(b_col + i) as usize] = (b[(b_col + i) as usize].clone() - sum) / a[(i * lda + i) as usize].clone();
            }
        }
    }
}

impl XsyevrScalar for MpfrFloat {
    fn xsyevr(
        jobz: u8, _range: u8, uplo: u8, n: i32, a: &mut [Self], lda: i32, _vl: Self, _vu: Self, _il: i32, _iu: i32, 
        _abstol: Self, m: &mut i32, w: &mut [Self], z: &mut [Self], ldz: i32, _isuppz: &mut [i32], 
        _work: &mut [Self], _lwork: i32, _iwork: &mut [i32], _liwork: i32, info: &mut i32,
    ) {
        *info = 0;
        *m = n;
        let is_u = uplo == b'U' || uplo == b'u';
        let want_z = jobz == b'V' || jobz == b'v';
        
        let n_us = n as usize;
        let mut mat = vec![MpfrFloat::zero(); n_us * n_us];
        for j in 0..n_us {
            for i in 0..n_us {
                let src_idx = if (is_u && i <= j) || (!is_u && i >= j) {
                    j * lda as usize + i
                } else {
                    i * lda as usize + j
                };
                mat[j * n_us + i] = a[src_idx].clone();
            }
        }
        
        if want_z {
            for j in 0..n_us {
                for i in 0..n_us {
                    z[j * ldz as usize + i] = if i == j { MpfrFloat::one() } else { MpfrFloat::zero() };
                }
            }
        }
        
        let tol = MpfrFloat::epsilon() * MpfrFloat::from_f64(1e3).unwrap();
        let mut iters = 0;
        loop {
            let mut max_off = MpfrFloat::zero();
            let mut p = 0;
            let mut q = 0;
            for j in 0..n_us {
                for i in 0..j {
                    let val = mat[j * n_us + i].clone().abs();
                    if val > max_off {
                        max_off = val;
                        p = i;
                        q = j;
                    }
                }
            }
            if max_off < tol || iters > 100 * n * n { break; }
            iters += 1;
            
            let app = mat[p * n_us + p].clone();
            let aqq = mat[q * n_us + q].clone();
            let apq = mat[q * n_us + p].clone();
            
            let theta = (aqq.clone() - app.clone()) / (MpfrFloat::from_f64(2.0).unwrap() * apq.clone());
            let t = if theta >= MpfrFloat::zero() {
                MpfrFloat::one() / (theta.clone() + (MpfrFloat::one() + theta.clone() * theta).sqrt())
            } else {
                MpfrFloat::from_f64(-1.0).unwrap() / (-theta.clone() + (MpfrFloat::one() + theta.clone() * theta).sqrt())
            };
            
            let c = MpfrFloat::one() / (MpfrFloat::one() + t.clone() * t.clone()).sqrt();
            let s = t * c.clone();
            
            for i in 0..n_us {
                if i != p && i != q {
                    let aip = mat[p * n_us + i].clone();
                    let aiq = mat[q * n_us + i].clone();
                    mat[p * n_us + i] = c.clone() * aip.clone() - s.clone() * aiq.clone();
                    mat[i * n_us + p] = mat[p * n_us + i].clone();
                    mat[q * n_us + i] = s.clone() * aip + c.clone() * aiq.clone();
                    mat[i * n_us + q] = mat[q * n_us + i].clone();
                }
            }
            mat[p * n_us + p] = c.clone() * c.clone() * app.clone() - MpfrFloat::from_f64(2.0).unwrap() * s.clone() * c.clone() * apq.clone() + s.clone() * s.clone() * aqq.clone();
            mat[q * n_us + q] = s.clone() * s.clone() * app + MpfrFloat::from_f64(2.0).unwrap() * s.clone() * c.clone() * apq + c.clone() * c.clone() * aqq;
            mat[q * n_us + p] = MpfrFloat::zero();
            mat[p * n_us + q] = MpfrFloat::zero();
            
            if want_z {
                for i in 0..n_us {
                    let zip = z[p * ldz as usize + i].clone();
                    let ziq = z[q * ldz as usize + i].clone();
                    z[p * ldz as usize + i] = c.clone() * zip.clone() - s.clone() * ziq.clone();
                    z[q * ldz as usize + i] = s.clone() * zip + c.clone() * ziq;
                }
            }
        }
        
        let mut idxs: Vec<usize> = (0..n_us).collect();
        idxs.sort_by(|&i, &j| mat[i * n_us + i].partial_cmp(&mat[j * n_us + j]).unwrap_or(Ordering::Equal));
        
        for (new_i, &old_i) in idxs.iter().enumerate() {
            w[new_i] = mat[old_i * n_us + old_i].clone();
        }
        
        if want_z {
            let mut z_new = vec![MpfrFloat::zero(); n_us * n_us];
            for (new_i, &old_i) in idxs.iter().enumerate() {
                for r in 0..n_us {
                    z_new[new_i * ldz as usize + r] = z[old_i * ldz as usize + r].clone();
                }
            }
            for j in 0..n_us {
                for i in 0..n_us {
                    z[j * ldz as usize + i] = z_new[j * ldz as usize + i].clone();
                }
            }
        }
    }
}

// ---------------------------------------------------------------------
// SVD: one-sided (Hestenes) Jacobi, at full MPFR working precision.
//
// Only the economy ("S") form used by `SVDEngine::factorblas` is
// supported: on exit `s` holds the k = min(m, n) singular values in
// descending order, `u` the m×k left factor (ld = ldu) and `vt` the k×n
// right factor (ld = ldvt). The LAPACK workspace-query protocol
// (lwork = -1 writes the required size to work[0]) is honoured; no
// workspace is actually needed. Both ?gesdd and ?gesvd dispatch here.
// ---------------------------------------------------------------------

fn jacobi_svd(
    m: usize, n: usize, a: &[MpfrFloat], lda: usize,
    s: &mut [MpfrFloat], u: &mut [MpfrFloat], ldu: usize,
    vt: &mut [MpfrFloat], ldvt: usize,
) -> i32 {
    // Work on G = A (m >= n) or G = Aᵀ (m < n), so G is r×c with r >= c.
    // Then G = Ug Σ Vgᵀ and A = Ug Σ Vgᵀ or A = Vg Σ Ugᵀ respectively.
    let trans = m < n;
    let (r, c) = if trans { (n, m) } else { (m, n) };
    let get = |i: usize, j: usize| if trans { a[i * lda + j].clone() } else { a[j * lda + i].clone() };
    // columns of G, stored column-major r×c
    let mut g: Vec<MpfrFloat> = (0..c).flat_map(|j| (0..r).map(move |i| (i, j))).map(|(i, j)| get(i, j)).collect();
    let mut v: Vec<MpfrFloat> = (0..c * c).map(|k| if k % (c + 1) == 0 { MpfrFloat::one() } else { MpfrFloat::zero() }).collect();

    let eps = MpfrFloat::epsilon();
    let max_sweeps = 30 + 4 * (super::default_precision() as usize / 53 + 1);
    let mut converged = c < 2;
    let dot = |x: &[MpfrFloat], y: &[MpfrFloat]| {
        x.iter().zip(y).fold(MpfrFloat::zero(), |acc, (xi, yi)| acc + xi.clone() * yi.clone())
    };
    for _ in 0..max_sweeps {
        if converged { break; }
        let mut rotated = false;
        for p in 0..c {
            for q in (p + 1)..c {
                let (gp, gq) = (&g[p * r..(p + 1) * r], &g[q * r..(q + 1) * r]);
                let alpha = dot(gp, gp);
                let beta = dot(gq, gq);
                let gamma = dot(gp, gq);
                if gamma.clone().abs() <= eps.clone() * (alpha.clone() * beta.clone()).sqrt() {
                    continue;
                }
                rotated = true;
                let two = MpfrFloat::from(2.0);
                let zeta = (beta - alpha) / (two * gamma);
                let one = MpfrFloat::one();
                let t = if zeta >= MpfrFloat::zero() {
                    one.clone() / (zeta.clone() + (one.clone() + zeta.clone() * zeta).sqrt())
                } else {
                    -one.clone() / (-zeta.clone() + (one.clone() + zeta.clone() * zeta).sqrt())
                };
                let cs = one.clone() / (one + t.clone() * t.clone()).sqrt();
                let sn = cs.clone() * t;
                for (mat, len) in [(&mut g, r), (&mut v, c)] {
                    for i in 0..len {
                        let xp = mat[p * len + i].clone();
                        let xq = mat[q * len + i].clone();
                        mat[p * len + i] = cs.clone() * xp.clone() - sn.clone() * xq.clone();
                        mat[q * len + i] = sn.clone() * xp + cs.clone() * xq;
                    }
                }
            }
        }
        converged = !rotated;
    }

    // singular values = column norms of G; sort descending
    let norms: Vec<MpfrFloat> = (0..c).map(|j| dot(&g[j * r..(j + 1) * r], &g[j * r..(j + 1) * r]).sqrt()).collect();
    let mut idx: Vec<usize> = (0..c).collect();
    idx.sort_by(|&i, &j| norms[j].partial_cmp(&norms[i]).unwrap_or(Ordering::Equal));
    let smax = if c > 0 { norms[idx[0]].clone() } else { MpfrFloat::zero() };
    let tiny = eps.clone() * smax * MpfrFloat::from(r as f64);

    // left vectors of G (r×c): normalise columns; complete null directions
    let mut ug: Vec<MpfrFloat> = vec![MpfrFloat::zero(); r * c];
    for (k, &j) in idx.iter().enumerate() {
        s[k] = norms[j].clone();
        if norms[j] > tiny {
            for i in 0..r { ug[k * r + i] = g[j * r + i].clone() / norms[j].clone(); }
        } else {
            // Gram-Schmidt a unit basis vector against the columns so far
            for e in 0..r {
                let mut col: Vec<MpfrFloat> = (0..r).map(|i| if i == e { MpfrFloat::one() } else { MpfrFloat::zero() }).collect();
                for _ in 0..2 {
                    for kk in 0..k {
                        let prev = &ug[kk * r..(kk + 1) * r];
                        let d = dot(prev, &col);
                        for i in 0..r { col[i] = col[i].clone() - d.clone() * prev[i].clone(); }
                    }
                }
                let nrm = dot(&col, &col).sqrt();
                if nrm > MpfrFloat::from(0.5) {
                    for i in 0..r { ug[k * r + i] = col[i].clone() / nrm.clone(); }
                    break;
                }
            }
        }
    }

    // scatter: A = Ug Σ Vgᵀ (no transpose) or A = Vg Σ Ugᵀ (transposed)
    for (k, &j) in idx.iter().enumerate() {
        for i in 0..m {
            u[k * ldu + i] = if trans { v[j * c + i].clone() } else { ug[k * r + i].clone() };
        }
        for i in 0..n {
            vt[i * ldvt + k] = if trans { ug[k * r + i].clone() } else { v[j * c + i].clone() };
        }
    }
    if converged { 0 } else { 1 }
}

fn jacobi_svd_entry(
    jobz: u8, m: i32, n: i32, a: &mut [MpfrFloat], lda: i32, s: &mut [MpfrFloat],
    u: &mut [MpfrFloat], ldu: i32, vt: &mut [MpfrFloat], ldvt: i32,
    work: &mut [MpfrFloat], lwork: i32, info: &mut i32,
) {
    *info = 0;
    if lwork == -1 {
        if let Some(w0) = work.first_mut() { *w0 = MpfrFloat::one(); }
        return;
    }
    assert!(jobz == b'S' || jobz == b's', "MPFR SVD supports only economy (job = 'S') output");
    *info = jacobi_svd(
        m as usize, n as usize, a, lda as usize, s, u, ldu as usize, vt, ldvt as usize,
    );
}

impl XgesddScalar for MpfrFloat {
    fn xgesdd(jobz: u8, m: i32, n: i32, a: &mut [Self], lda: i32, s: &mut [Self], u: &mut [Self], ldu: i32, vt: &mut [Self], ldvt: i32, work: &mut [Self], lwork: i32, _iwork: &mut [i32], info: &mut i32) {
        jacobi_svd_entry(jobz, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, info)
    }
}
impl XgesvdScalar for MpfrFloat {
    fn xgesvd(jobu: u8, jobvt: u8, m: i32, n: i32, a: &mut [Self], lda: i32, s: &mut [Self], u: &mut [Self], ldu: i32, vt: &mut [Self], ldvt: i32, work: &mut [Self], lwork: i32, info: &mut i32) {
        assert_eq!(jobu, jobvt, "MPFR SVD requires jobu == jobvt");
        jacobi_svd_entry(jobu, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, info)
    }
}
impl XgemmScalar for MpfrFloat {
    fn xgemm(transa: u8, transb: u8, m: i32, n: i32, k: i32, alpha: Self, a: &[Self], lda: i32, b: &[Self], ldb: i32, beta: Self, c: &mut [Self], ldc: i32) {
        let is_ta = transa == b'T' || transa == b't' || transa == b'C' || transa == b'c';
        let is_tb = transb == b'T' || transb == b't' || transb == b'C' || transb == b'c';
        let get_a = |i: i32, j: i32| if is_ta { a[(i * lda + j) as usize].clone() } else { a[(j * lda + i) as usize].clone() };
        let get_b = |i: i32, j: i32| if is_tb { b[(i * ldb + j) as usize].clone() } else { b[(j * ldb + i) as usize].clone() };
        for j in 0..n {
            for i in 0..m {
                let mut sum = MpfrFloat::zero();
                for l in 0..k { sum = sum + get_a(i, l) * get_b(l, j); }
                let idx = (j * ldc + i) as usize;
                c[idx] = alpha.clone() * sum + beta.clone() * c[idx].clone();
            }
        }
    }
}
impl XgemvScalar for MpfrFloat {
    fn xgemv(trans: u8, m: i32, n: i32, alpha: Self, a: &[Self], lda: i32, x: &[Self], incx: i32, beta: Self, y: &mut [Self], incy: i32) {
        let is_t = trans == b'T' || trans == b't' || trans == b'C' || trans == b'c';
        let get_x = |i: i32| x[(i * incx) as usize].clone();
        let get_a = |i: i32, j: i32| a[(j * lda + i) as usize].clone();
        if !is_t {
            for i in 0..m {
                let mut sum = MpfrFloat::zero();
                for j in 0..n { sum = sum + get_a(i, j) * get_x(j); }
                let idx = (i * incy) as usize;
                y[idx] = alpha.clone() * sum + beta.clone() * y[idx].clone();
            }
        } else {
            for j in 0..n {
                let mut sum = MpfrFloat::zero();
                for i in 0..m { sum = sum + get_a(i, j) * get_x(i); }
                let idx = (j * incy) as usize;
                y[idx] = alpha.clone() * sum + beta.clone() * y[idx].clone();
            }
        }
    }
}
impl XsymvScalar for MpfrFloat {
    fn xsymv(uplo: u8, n: i32, alpha: Self, a: &[Self], lda: i32, x: &[Self], incx: i32, beta: Self, y: &mut [Self], incy: i32) {
        let is_u = uplo == b'U' || uplo == b'u';
        let get_x = |i: i32| x[(i * incx) as usize].clone();
        let get_a = |i: i32, j: i32| {
            if i <= j { if is_u { a[(j * lda + i) as usize].clone() } else { a[(i * lda + j) as usize].clone() } }
            else { if is_u { a[(i * lda + j) as usize].clone() } else { a[(j * lda + i) as usize].clone() } }
        };
        for i in 0..n {
            let mut sum = MpfrFloat::zero();
            for j in 0..n { sum = sum + get_a(i, j) * get_x(j); }
            let idx = (i * incy) as usize;
            y[idx] = alpha.clone() * sum + beta.clone() * y[idx].clone();
        }
    }
}
impl XsyrkScalar for MpfrFloat {
    fn xsyrk(uplo: u8, trans: u8, n: i32, k: i32, alpha: Self, a: &[Self], lda: i32, beta: Self, c: &mut [Self], ldc: i32) {
        let is_u = uplo == b'U' || uplo == b'u';
        let is_t = trans == b'T' || trans == b't' || trans == b'C' || trans == b'c';
        let get_a = |i: i32, j: i32| if is_t { a[(i * lda + j) as usize].clone() } else { a[(j * lda + i) as usize].clone() };
        for j in 0..n {
            for i in 0..n {
                if (is_u && i > j) || (!is_u && i < j) { continue; }
                let mut sum = MpfrFloat::zero();
                for l in 0..k { sum = sum + get_a(i, l) * get_a(j, l); }
                let idx = (j * ldc + i) as usize;
                c[idx] = alpha.clone() * sum + beta.clone() * c[idx].clone();
            }
        }
    }
}
impl Xsyr2kScalar for MpfrFloat {
    fn xsyr2k(uplo: u8, trans: u8, n: i32, k: i32, alpha: Self, a: &[Self], lda: i32, b: &[Self], ldb: i32, beta: Self, c: &mut [Self], ldc: i32) {
        let is_u = uplo == b'U' || uplo == b'u';
        let is_t = trans == b'T' || trans == b't' || trans == b'C' || trans == b'c';
        let get_a = |i: i32, j: i32| if is_t { a[(i * lda + j) as usize].clone() } else { a[(j * lda + i) as usize].clone() };
        let get_b = |i: i32, j: i32| if is_t { b[(i * ldb + j) as usize].clone() } else { b[(j * ldb + i) as usize].clone() };
        for j in 0..n {
            for i in 0..n {
                if (is_u && i > j) || (!is_u && i < j) { continue; }
                let mut sum = MpfrFloat::zero();
                for l in 0..k { sum = sum + get_a(i, l) * get_b(j, l) + get_b(i, l) * get_a(j, l); }
                let idx = (j * ldc + i) as usize;
                c[idx] = alpha.clone() * sum + beta.clone() * c[idx].clone();
            }
        }
    }
}
