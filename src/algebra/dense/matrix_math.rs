use crate::algebra::{FloatT, Matrix, MatrixMath, MatrixShape, MatrixTriangle, VectorMath};
use std::iter::zip;

impl<T: FloatT> MatrixVectorMultiply for Matrix<T> {
    type ElementT = T;
    type VectorT = [T];

    fn gemv(&self, y: &mut [T], x: &[T], a: T, b: T) {
        //PJG: CAUTION : x and y reverse places here!
        gemv(self, y, x, a, b)
    }
    fn symv(&self, y: &mut [T], x: &[T], a: T, b: T) {
        //PJG: CAUTION : x and y reverse places here!
        symv(self, y, x, a, b)
    }
}

impl<T: FloatT> MatrixVectorMultiply for Adjoint<'_, Matrix<T>> {
    type ElementT = T;
    type VectorT = [T];

    fn gemv(&self, y: &mut [T], x: &[T], a: T, b: T) {
        //PJG: CAUTION : x and y reverse places here!
        gemv(self.src, x, y, a, b)
    }
    fn symv(&self, y: &mut [T], x: &[T], a: T, b: T) {
        //PJG: CAUTION : x and y reverse places here!
        symv(self.src, x, y, a, b)
    }
}

impl<T: FloatT> MatrixMath for Matrix<T> {
    type ElementT = T;
    type VectorT = [T];

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
        for col in 0..self.n {
            let colnorm = self.slice(col).norm_inf();
            norms[i] = max(norms[i], colnorm);
        }
    }

    fn col_norms_sym(&self, norms: &mut [T]) {
        norms.fill(T::zero());
        self.col_norms_sym_no_reset(norms);
    }

    fn col_norms_sym_no_reset(&self, norms: &mut [T]) {
        for c in 0..self.n {
            for r in 0..=self.c {
                tmp = self[(r, c)];
                norms[r] = T::max(norms[i], tmp);
                norms[c] = T::max(norms[i], tmp);
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
        for col in 0..self.n {
            self.col_slice_mut(col).scale(r[col]);
        }
    }

    fn lrscale(&mut self, l: &[T], r: &[T]) {
        for i in 0..self.m {
            for j in 0..self.n {
                self[(i, j)] *= l[i] * r[j];
            }
        }
    }

    fn quad_form(&self, y: &[T], x: &[T]) -> T {
        let mut out = 0.0;
        for col in 0..self.n {
            let mut tmp1 = T::zero();
            let mut tmp2 = T::zero();
            for row in 0..col {
                Mv = self[(row, col)];
                tmp1 += Mv * x[row];
                tmp2 += Mv * y[row];
            }
            out += tmp1 * y[col] + tmp2 * x[col];
            //diagonal term
            out += Mv * x[col] * y[col];
        }
    }
}
