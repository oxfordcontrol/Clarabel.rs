use crate::{MatrixShape,MatrixTriangle};

// All internal math for all solver implementations should go
// through these core traits, which are implemented generically
// for floats of type FloatT.   

pub trait ScalarMath<T> {
    fn clip(s: T, min_thresh: T, max_thresh: T, min_new: T, max_new: T) -> T;
}

pub trait VectorMath<T> {
    //copy values from another vector
    fn copy_from(&mut self, src: &Self);

    //generic scalar elementwise operations.
    fn scalarop(&mut self, op: impl Fn(T) -> T);
    fn scalarop_from(&mut self, op: impl Fn(T) -> T, v: &Self);

    //elementwise mut operations
    fn translate(&mut self, c: T);
    fn scale(&mut self, c: T);
    fn reciprocal(&mut self);
    fn sqrt(&mut self);
    fn rsqrt(&mut self); //reciprocal sqrt
    fn negate(&mut self);

    fn hadamard(&mut self, y: &Self);
    fn clip(&mut self, min_thresh: T, max_thresh: T, min_new: T, max_new: T);

    //norms and friends
    fn dot(&self, y: &Self) -> T;
    fn sumsq(&self) -> T;
    fn norm(&self) -> T;
    fn norm_scaled(&self, v: &Self) -> T;
    fn norm_inf(&self) -> T;
    fn norm_one(&self) -> T;

    //stats
    fn minimum(&self) -> T;
    fn maximum(&self) -> T;
    fn mean(&self) -> T;

    //blas-like vector ops
    fn axpby(&mut self, a: T, x: &Self, b: T); //self = a*x+b*self
    fn waxpby(&mut self, a: T, x: &Self, b: T, y: &Self); //self = a*x+b*y
}

pub trait MatrixMath<T, V: ?Sized> {
    //matrix properties
    fn nrows(&self) -> usize;
    fn ncols(&self) -> usize;
    fn nnz(&self) -> usize;
    fn is_square(&self) -> bool;

    //inf norms of rows and columns
    fn col_norms(&self, norms: &mut V);
    fn col_norms_no_reset(&self, norms: &mut V);
    fn col_norms_sym(&self, norms: &mut V);
    fn col_norms_sym_no_reset(&self, norms: &mut V);
    fn row_norms(&self, norms: &mut V);
    fn row_norms_no_reset(&self, norms: &mut V);

    //scalar mut operations
    fn scale(&mut self, c: T);

    //left and right multiply by diagonals
    fn lscale(&mut self, l: &V);
    fn rscale(&mut self, r: &V);
    fn lrscale(&mut self, l: &V, r: &V);

    // general matrix-vector multiply, blas like
    // y = a*self*x + b*y
    fn gemv(&self, y: &mut V, trans: MatrixShape, x: &V, a: T, b: T);

    // symmetric matrix-vector multiply, blas like
    // y = a*self*x + b*y.  The matrix should be
    //in either triu or tril form, with the other
    //half of the triangle assumed symmetric
    fn symv(&self, y: &mut [T], tri: MatrixTriangle, x: &[T], a: T, b: T);

    // quadratic form for a symmetric matrix.  Assumes upper
    // triangular form for the matrix
    fn quad_form(&self, y: &[T], x: &[T]) -> T;
}
