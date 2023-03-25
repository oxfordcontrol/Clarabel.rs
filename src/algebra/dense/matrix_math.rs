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

    // PJG: this really wants a unit test because I don't
    //understand why I have to fix Mv = 0. to start.
    fn quad_form(&self, y: &[T], x: &[T]) -> T {
        let mut out = T::zero();
        for col in 0..self.n {
            let mut tmp1 = T::zero();
            let mut tmp2 = T::zero();
            let mut Mv = T::zero();
            for row in 0..col {
                Mv = self[(row, col)];
                tmp1 += Mv * x[row];
                tmp2 += Mv * y[row];
            }
            out += tmp1 * y[col] + tmp2 * x[col];
            //diagonal term
            out += Mv * x[col] * y[col];
        }
        out
    }
}
