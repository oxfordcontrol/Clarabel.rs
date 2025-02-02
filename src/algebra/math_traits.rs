// All internal math for all solver implementations should go
// through these core traits, which are implemented generically
// for floats of type FloatT.

/// Scalar operations on [`FloatT`](crate::algebra::FloatT)
pub trait ScalarMath {
    /// Applies a threshold value.   
    ///
    /// Restricts the value to be at least `min_thresh` and at most `max_thresh`.
    fn clip(&self, min_thresh: Self, max_thresh: Self) -> Self;

    /// Safe calculation for log barriers.
    ///
    /// Returns log(s) if s > 0   -Infinity otherwise.
    fn logsafe(&self) -> Self;
}

/// Vector operations on slices of [`FloatT`](crate::algebra::FloatT)
pub trait VectorMath<T> {
    /// Copy values from `src` to `self`
    fn copy_from(&mut self, src: &Self) -> &mut Self;

    /// Make a new vector from a subset of elements
    fn select(&self, index: &[bool]) -> Vec<T>;

    /// Apply an elementwise operation on a vector.
    fn scalarop(&mut self, op: impl Fn(T) -> T) -> &mut Self;

    /// Apply an elementwise operation to `v` and assign the
    /// results to `self`.
    fn scalarop_from(&mut self, op: impl Fn(T) -> T, v: &Self) -> &mut Self;

    /// Elementwise translation.
    fn translate(&mut self, c: T) -> &mut Self;

    /// set all elements to the same value
    fn set(&mut self, c: T) -> &mut Self;

    /// Elementwise scaling.
    fn scale(&mut self, c: T) -> &mut Self;

    /// Elementwise reciprocal.
    fn recip(&mut self) -> &mut Self;

    /// Elementwise square root.
    fn sqrt(&mut self) -> &mut Self;

    /// Elementwise inverse square root.
    fn rsqrt(&mut self) -> &mut Self;

    /// Elementwise negation of entries.
    fn negate(&mut self) -> &mut Self;

    /// Elementwise scaling by another vector. Produces `self[i] = self[i] * y[i]`
    fn hadamard(&mut self, y: &Self) -> &mut Self;

    /// Vector version of [clip](crate::algebra::ScalarMath::clip)
    fn clip(&mut self, min_thresh: T, max_thresh: T) -> &mut Self;

    /// Normalize, returning the norm.  Do nothing if norm == 0.  
    fn normalize(&mut self) -> T;

    /// Dot product
    fn dot(&self, y: &Self) -> T;

    /// computes dot(z + αdz,s + αds) without intermediate allocation
    fn dot_shifted(z: &[T], s: &[T], dz: &[T], ds: &[T], α: T) -> T;

    /// Standard Euclidian or 2-norm distance from `self` to `y`
    fn dist(&self, y: &Self) -> T;

    /// Sum of elements.
    fn sum(&self) -> T;

    /// Sum of squares of the elements.
    fn sumsq(&self) -> T;

    /// 2-norm
    fn norm(&self) -> T;

    /// Infinity norm
    fn norm_inf(&self) -> T;

    /// One norm
    fn norm_one(&self) -> T;

    /// 2-norm of an elementwise scaling of `self` by `v`
    fn norm_scaled(&self, v: &Self) -> T;

    /// Inf-norm of an elementwise scaling of `self` by `v`
    fn norm_inf_scaled(&self, v: &Self) -> T;

    /// Inf-norm of an elementwise scaling of `self` by `v`
    fn norm_one_scaled(&self, v: &Self) -> T;

    /// Inf-norm of vector difference
    fn norm_inf_diff(&self, b: &Self) -> T;

    /// Minimum value in vector
    fn minimum(&self) -> T;

    /// Maximum value in vector
    fn maximum(&self) -> T;

    /// Mean value in vector
    fn mean(&self) -> T;

    /// Checks if all elements are finite, i.e. no Infs or NaNs
    fn is_finite(&self) -> bool;

    //blas-like vector ops
    //--------------------

    /// BLAS-like shift and scale in place.  Produces `self = a*x+b*self`
    fn axpby(&mut self, a: T, x: &Self, b: T) -> &mut Self;

    /// BLAS-like shift and scale, non in-place version.  Produces `self = a*x+b*y`
    fn waxpby(&mut self, a: T, x: &Self, b: T, y: &Self) -> &mut Self;
}

/// Matrix operations for matrices of [`FloatT`](crate::algebra::FloatT)
pub(crate) trait MatrixVectorMultiply<T> {
    /// BLAS-like general matrix-vector multiply.  Produces `y = a*self*x + b*y`
    fn gemv(&self, y: &mut [T], x: &[T], a: T, b: T);
}

pub(crate) trait SymMatrixVectorMultiply<T> {
    /// BLAS-like symmetric matrix-vector multiply.  Produces `y = a*self*x + b*y`.  
    /// The matrix source data should be triu.
    fn symv(&self, y: &mut [T], x: &[T], a: T, b: T);
}

/// Operations on matrices of [`FloatT`](crate::algebra::FloatT)
pub trait MatrixMath<T> {
    /// Compute columnwise sums on a matrix and
    /// assign the results to the vector `sums`
    fn col_sums(&self, sums: &mut [T]);

    /// Compute columnwise sums on a matrix and
    /// assign the results to the vector `sums`
    fn row_sums(&self, sums: &mut [T]);

    /// Compute columnwise infinity norm operations on
    /// a matrix and assign the results to the vector `norms`
    fn col_norms(&self, norms: &mut [T]);

    /// Compute columnwise infinity norm operations on
    /// a matrix and assign the results to the vector `norms`.
    /// In the `no_reset` version of this function, if `norms[i]`
    /// is already larger the norm of the $i^{th}$ columns, then
    /// its value is not changed
    fn col_norms_no_reset(&self, norms: &mut [T]);

    /// Compute columnwise infinity norm operations on
    /// a symmetric matrix
    fn col_norms_sym(&self, norms: &mut [T]);

    /// Compute columnwise infinity norm operations on
    /// a symmetric matrix without reset
    fn col_norms_sym_no_reset(&self, norms: &mut [T]);

    /// Compute rowwise infinity norm operations on
    /// a matrix and assign the results to the vector `norms`
    fn row_norms(&self, norms: &mut [T]);

    /// Compute rowwise infinity norm operations on
    /// a matrix and assign the results to the vector `norms`
    /// without reset
    fn row_norms_no_reset(&self, norms: &mut [T]);

    /// Quadratic form for a symmetric matrix.  Assumes that the
    /// matrix `M = self` is in upper triangular form, and produces
    /// `y^T*M*x`
    ///
    /// PJG: Maybe this should be on symmetric only.
    fn quad_form(&self, y: &[T], x: &[T]) -> T;
}

/// Operations on mutable matrices of [`FloatT`](crate::algebra::FloatT)
pub trait MatrixMathMut<T> {
    /// Elementwise scaling
    fn scale(&mut self, c: T);

    /// Elementwise negation
    fn negate(&mut self);

    /// Left multiply the matrix `self` by `Diagonal(l)`
    fn lscale(&mut self, l: &[T]);

    /// Right multiply the matrix self by `Diagonal(r)`
    fn rscale(&mut self, r: &[T]);

    /// Left and multiply the matrix self by diagonal matrices,
    /// producing `A = Diagonal(l)*A*Diagonal(r)`
    fn lrscale(&mut self, l: &[T], r: &[T]);
}
