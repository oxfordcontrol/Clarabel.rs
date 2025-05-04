#![allow(non_snake_case)]
use crate::algebra::*;


// PJG : MatrixMath<T> should be implemented for a more 
// general type, e.g. the DenseStorageMatrix<S,T> type 
// or similar.   That would provide math functionality
// for more types, e.g. statically size matrices or 
// the ones on borrowed data.   


impl<T: FloatT> MatrixMath<T> for Matrix<T> {

    fn col_sums(&self, sums: &mut [T]) {
        assert_eq!(self.ncols(), sums.len());
        for (col, sum) in sums.iter_mut().enumerate() {
            *sum = self.col_slice(col).sum();
        }
    }

    fn row_sums(&self, sums: &mut [T]) {
        assert_eq!(self.nrows(), sums.len());
        sums.fill(T::zero());
        for col in 0..self.ncols() {
            let slice = self.col_slice(col);
            for (row, &v) in slice.iter().enumerate() {
                sums[row] += v;
            }
        }
    }

    fn col_norms(&self, norms: &mut [T]) {
        norms.fill(T::zero());
        self.col_norms_no_reset(norms);
    }

    fn col_norms_no_reset(&self, norms: &mut [T]) {
        for (i, norm) in norms.iter_mut().enumerate() {
            let colnorm = self.col_slice(i).norm_inf();
            *norm = T::max(*norm, colnorm);
        }
    }

    fn col_norms_sym(&self, norms: &mut [T]) {
        norms.fill(T::zero());
        self.col_norms_sym_no_reset(norms);
    }

    fn col_norms_sym_no_reset(&self, norms: &mut [T]) {
        for c in 0..self.ncols() {
            for r in 0..=c {
                let tmp = self[(r, c)];
                norms[r] = T::max(norms[r], tmp);
                norms[c] = T::max(norms[c], tmp);
            }
        }
    }

    fn row_norms(&self, norms: &mut [T]) {
        norms.fill(T::zero());
        self.row_norms_no_reset(norms);
    }

    fn row_norms_no_reset(&self, norms: &mut [T]) {
        for r in 0..self.nrows() {
            for c in 0..self.ncols() {
                norms[r] = T::max(norms[r], T::abs(self[(r, c)]))
            }
        }
    }
}

impl<T: FloatT> MatrixMathMut<T> for Matrix<T> {

    //scalar mut operations
    fn scale(&mut self, c: T) {
        self.data.scale(c);
    }

    fn negate(&mut self) {
        self.data.negate();
    }

    fn lscale(&mut self, l: &[T]) {
        for col in 0..self.ncols() {
            self.col_slice_mut(col).hadamard(l);
        }
    }

    fn rscale(&mut self, r: &[T]) {
        for (col, val) in r.iter().enumerate() {
            self.col_slice_mut(col).scale(*val);
        }
    }

    fn lrscale(&mut self, l: &[T], r: &[T]) {
        for i in 0..self.nrows() {
            for j in 0..self.ncols() {
                self[(i, j)] *= l[i] * r[j];
            }
        }
    }

}

impl<T> Matrix<T>
where
    T: FloatT,
{
    #[allow(dead_code)]
    pub(crate) fn kron<MATA, MATB>(&mut self, A: &MATA, B: &MATB)
    where
        MATA: DenseMatrix<T>,
        MATB: DenseMatrix<T>,
    {
        let (pp, qq) = A.size();
        let (rr, ss) = B.size();
        assert!(self.nrows() == pp * rr);
        assert!(self.ncols() == qq * ss);

        let mut i = 0;
        for q in 0..qq {
            for s in 0..ss {
                for p in 0..pp {
                    let Apq = A[(p, q)];
                    for r in 0..rr {
                        self.data_mut()[i] = (Apq) * B[(r, s)];
                        i += 1;
                    }
                }
            }
        }
    }
}


// additional functions that require floating point operations

// allow dead code here since dense matrix and its supporting
// functionality could eventually become a public interface.
#[allow(dead_code)]
impl<S,T> DenseStorageMatrix<S, T> 
where 
    T: FloatT,
    S: AsRef<[T]> + AsMut<[T]>
{
    /// Set A = (A + A') / 2.  Assumes A is real
    pub fn symmetric_part(&mut self) -> &mut Self {
        assert!(self.is_square());
        let half: T = (0.5_f64).as_T();

        for r in 0..self.nrows() {
            for c in 0..r {
                let val = half * (self[(r, c)] + self[(c, r)]);
                self[(c, r)] = val;
                self[(r, c)] = val;
            }
        }
        self
    }
}



pub(crate) fn svec_to_mat<S,T>(M: &mut DenseStorageMatrix<S,T>, x: &[T]) 
where
    T: FloatT,
    S: AsRef<[T]> + AsMut<[T]>,
{
    let mut idx = 0;
    for col in 0..M.ncols() {
        for row in 0..=col {
            if row == col {
                M[(row, col)] = x[idx];
            } else {
                M[(row, col)] = x[idx] * T::FRAC_1_SQRT_2();
                M[(col, row)] = x[idx] * T::FRAC_1_SQRT_2();
            }
            idx += 1;
        }
    }
}

//PJG : Perhaps implementation for Symmetric type would be faster
pub(crate) fn mat_to_svec<T,MATM>(x: &mut [T], M: &MATM)
where 
MATM: DenseMatrix<T>,
     T: FloatT,
{
    let mut idx = 0;
    for col in 0..M.ncols() {
        for row in 0..=col {
            x[idx] = {
                if row == col {
                    M[(row, col)]
                } else {
                    (M[(row, col)] + M[(col, row)]) * T::FRAC_1_SQRT_2()
                }
            };
            idx += 1;
        }
    }
}


#[test]
fn test_row_col_sums_and_norms() {
    #[rustfmt::skip]
    let A = Matrix::from(&[
        [-1.,  4.,  6.], 
        [ 3., -8.,  7.], 
        [ 0.,  4.,  9.],
    ]);

    let mut rsums = vec![0.0; 3];
    let mut csums = vec![0.0; 3];

    A.row_sums(&mut rsums);
    assert_eq!(rsums, [9.0, 2.0, 13.0]);
    A.col_sums(&mut csums);
    assert_eq!(csums, [2.0, 0.0, 22.0]);

    let mut rnorms = vec![0.0; 3];
    let mut cnorms = vec![0.0; 3];

    A.row_norms(&mut rnorms);
    assert!(rnorms == [6.0, 8.0, 9.0]);
    A.col_norms(&mut cnorms);
    assert!(cnorms == [3.0, 8.0, 9.0]);

    //no reset versions
    let mut rnorms = vec![0.0; 3];
    let mut cnorms = vec![0.0; 3];
    rnorms[2] = 100.;
    cnorms[2] = 100.;

    A.row_norms_no_reset(&mut rnorms);
    assert!(rnorms == [6.0, 8.0, 100.0]);
    A.col_norms_no_reset(&mut cnorms);
    assert!(cnorms == [3.0, 8.0, 100.0]);
}

#[test]
#[rustfmt::skip]
fn test_l_r_scalings() {

    let A = Matrix::from(&[
        [-1.,  4.,  6.], 
        [ 3., -8.,  7.], 
        [ 0.,  4.,  9.],
    ]);

    let lscale = vec![1., -2., 3.];
    let rscale = vec![-2., 1., -3.];

    //right scale
    let mut B = A.clone();
    B.rscale(&rscale);
    let Btest = Matrix::from(&[
        [ 2.,  4.,  -18.], 
        [-6., -8.,  -21.], 
        [ 0.,  4.,  -27.],
    ]);
    assert_eq!(B,Btest);

    //left scale
    let mut B = A.clone();
    B.lscale(&lscale);
    let Btest = Matrix::from(&[
        [-1.,  4.,   6.], 
        [-6., 16., -14.], 
        [ 0., 12.,  27.],
    ]);
    assert_eq!(B,Btest);

    //left-right scale
    let mut B = A;
    B.lrscale(&lscale, &rscale);
    let Btest = Matrix::from(&[
        [ 2.,  4., -18.], 
        [12., 16.,  42.], 
        [ 0., 12., -81.],
    ]);
    assert_eq!(B,Btest);
}

#[test]
fn test_symmetric_part() {

    let mut A = Matrix::from(&[
        [-1.,  4.,  6.],
        [ 2., -8.,  8.],
        [ 0.,  4.,  9.],
    ]);

    let B = Matrix::from(&[
        [-1.,  3.,  3.],
        [ 3., -8.,  6.],
        [ 3.,  6.,  9.],
    ]);

    A.symmetric_part();
    assert_eq!(B,A);
}

#[test]
fn test_col_norms_sym() {

    let A = Matrix::from(&[
        [-1.,  4.,  6.],
        [ 2., -8.,  8.],
        [ 0.,  4.,  9.],
    ]);

    let mut v = vec![0.0; 3];
    A.col_norms_sym(&mut v);
    assert_eq!(v, [6.0, 8.0, 9.0]);
}



#[test]
#[rustfmt::skip]
fn test_kron() {

    let A = Matrix::from(
        &[[ 1.,  2.],
          [ 4.,  5.]]);

    let B = Matrix::from(
        &[[ 1.,  2.]]);


    // A ⊗ B
    let (k1, m1) = A.size();
    let (k2, m2) = B.size();
    let mut K = Matrix::<f64>::zeros((k1 * k2, m1 * m2));
    K.kron(&A, &B);

    let Ktest = Matrix::from(
        &[[ 1.,  2.,  2.,  4.],
          [ 4.,  8.,  5., 10.]]);

    assert_eq!(K,Ktest);

    // A' ⊗ B
    let (k1, m1) = A.t().size();
    let (k2, m2) = B.size();
    let mut K = Matrix::<f64>::zeros((k1 * k2, m1 * m2));
    K.kron(&A.t(), &B);

    let Ktest = Matrix::from(
        &[[ 1.,  2.,  4.,  8.],
          [ 2.,  4.,  5., 10.]]);
          
    assert_eq!(K,Ktest);

    // A ⊗ B'            
    let (k1, m1) = A.size();
    let (k2, m2) = B.t().size();
    let mut K = Matrix::<f64>::zeros((k1 * k2, m1 * m2));
    K.kron(&A, &B.t());

    let Ktest = Matrix::from(
        &[[1., 2. ],
          [2., 4. ],
          [4., 5. ],
          [8., 10.]]);
          
    assert_eq!(K,Ktest);

    // A' ⊗ B'  
    let (k1, m1) = A.t().size();
    let (k2, m2) = B.t().size();  
    let mut K = Matrix::<f64>::zeros((k1 * k2, m1 * m2));  
    K.kron(&A.t(), &B.t());

    let Ktest = Matrix::from(
        &[[1., 4. ],
          [2., 8. ],
          [2., 5. ],
          [4., 10.]]);

    assert_eq!(K,Ktest);
}



#[test]
fn test_svec_conversions() {
    let n = 3;

    let X = Matrix::from(&[
        [1., 3., -2.], //
        [3., -4., 7.], //
        [-2., 7., 5.], //
    ]);

    let Y = Matrix::from(&[
        [2., 5., -4.],  //
        [5., 6., 2.],   //
        [-4., 2., -3.], //
    ]);

    let mut Z = Matrix::zeros((3, 3));

    let mut x = vec![0.; triangular_number(n)];
    let mut y = vec![0.; triangular_number(n)];

    // check inner product identity
    mat_to_svec(&mut x, &X);
    mat_to_svec(&mut y, &Y);

    assert!(f64::abs(x.dot(&y) - X.data().dot(Y.data())) < 1e-12);

    // check round trip
    mat_to_svec(&mut x, &X);
    svec_to_mat(&mut Z, &x);
    assert!(X.data().norm_inf_diff(Z.data()) < 1e-12);
}
