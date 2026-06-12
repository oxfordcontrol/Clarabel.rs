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

impl XgesddScalar for MpfrFloat {
    fn xgesdd(_jobz: u8, _m: i32, _n: i32, _a: &mut [Self], _lda: i32, _s: &mut [Self], _u: &mut [Self], _ldu: i32, _vt: &mut [Self], _ldvt: i32, _work: &mut [Self], _lwork: i32, _iwork: &mut [i32], _info: &mut i32) { unimplemented!() }
}
impl XgesvdScalar for MpfrFloat {
    fn xgesvd(_jobu: u8, _jobvt: u8, _m: i32, _n: i32, _a: &mut [Self], _lda: i32, _s: &mut [Self], _u: &mut [Self], _ldu: i32, _vt: &mut [Self], _ldvt: i32, _work: &mut [Self], _lwork: i32, _info: &mut i32) { unimplemented!() }
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
