#![allow(non_snake_case)]
use crate::algebra::{FloatT, Matrix, MatrixMath, VectorMath};

impl<T: FloatT> MatrixMath for Matrix<T> {
    type T = T;

    //scalar mut operations
    fn scale(&mut self, c: T) {
        self.data.scale(c);
    }

    fn negate(&mut self) {
        self.data.negate();
    }

    fn col_norms(&self, norms: &mut [T]) {
        norms.fill(T::zero());
        self.col_norms_no_reset(norms);
    }

    fn col_norms_no_reset(&self, norms: &mut [T]) {
        for (i, norm) in norms.iter_mut().enumerate() {
            let colnorm = self.col_slice(i).norm_inf();
            *norm = Self::T::max(*norm, colnorm);
        }
    }

    fn col_norms_sym(&self, norms: &mut [T]) {
        norms.fill(T::zero());
        self.col_norms_sym_no_reset(norms);
    }

    fn col_norms_sym_no_reset(&self, norms: &mut [T]) {
        for c in 0..self.n {
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
        for r in 0..self.m {
            for c in 0..self.n {
                norms[r] = T::max(norms[r], T::abs(self[(r, c)]))
            }
        }
    }

    fn lscale(&mut self, l: &[T]) {
        for col in 0..self.n {
            self.col_slice_mut(col).hadamard(l);
        }
    }

    fn rscale(&mut self, r: &[T]) {
        for (col, val) in r.iter().enumerate() {
            self.col_slice_mut(col).scale(*val);
        }
    }

    fn lrscale(&mut self, l: &[T], r: &[T]) {
        for i in 0..self.m {
            for j in 0..self.n {
                self[(i, j)] *= l[i] * r[j];
            }
        }
    }

    // PJG: this should probably only be implemented as part of
    // some SymmetricMatrixMath trait.  Uses upper triangle only
    // and assumes that the rest is symmetric.
    fn quad_form(&self, y: &[T], x: &[T]) -> T {
        assert_eq!(self.m, self.n);
        let mut out = T::zero();
        for col in 0..self.n {
            let mut tmp1 = T::zero();
            let mut tmp2 = T::zero();
            for row in 0..=col {
                let Mv = self[(row, col)];
                if row < col {
                    tmp1 += Mv * x[row];
                    tmp2 += Mv * y[row];
                } else {
                    //diagonal term
                    out += Mv * x[col] * y[col];
                }
            }
            out += tmp1 * y[col] + tmp2 * x[col];
        }
        out
    }
}

#[test]
fn test_quad_form() {
    let mut A = Matrix::from(&[
        [1., 4.], //
        [4., 5.], //
    ]);

    let x = vec![1.0, 2.0];
    let y = vec![3.0, 4.0];
    assert!(A.quad_form(&x, &y) == 83.0);

    //remove lower triangle part and check again.
    //should not change the result.
    A[(1, 0)] = 0.0;
    assert!(A.quad_form(&x, &y) == 83.0);
}

#[test]
fn test_row_col_norms() {
    #[rustfmt::skip]
    let A = Matrix::from(&[
        [-1.,  4.,  6.], 
        [ 3., -8.,  7.], 
        [ 0.,  4.,  9.],
    ]);

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
    let mut B = A.clone();
    B.lrscale(&lscale, &rscale);
    let Btest = Matrix::from(&[
        [ 2.,  4., -18.], 
        [12., 16.,  42.], 
        [ 0., 12., -81.],
    ]);
    assert_eq!(B,Btest);
}
