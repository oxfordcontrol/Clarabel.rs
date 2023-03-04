use super::*;

impl<T: FloatT> ScalarMath<T> for T {
    fn clip(&self, min_thresh: T, max_thresh: T, min_new: T, max_new: T) -> T {
        if *self < min_thresh {
            min_new
        } else if *self > max_thresh {
            max_new
        } else {
            *self
        }
    }

    fn logsafe(&self) -> T {
        if *self <= T::zero() {
            -T::infinity()
        } else {
            self.ln()
        }
    }
}

impl<T: FloatT> VectorMath<T> for [T] {
    fn copy_from(&mut self, src: &[T]) -> &mut Self {
        self.copy_from_slice(src);
        self
    }

    fn select(&self, index: &[bool]) -> Vec<T> {
        assert_eq!(self.len(), index.len());
        self.iter()
            .zip(index)
            .filter(|(_x, &b)| b)
            .map(|(&x, _b)| x)
            .collect()
    }

    fn scalarop(&mut self, op: impl Fn(T) -> T) -> &mut Self {
        for x in &mut *self {
            *x = op(*x);
        }
        self
    }

    fn scalarop_from(&mut self, op: impl Fn(T) -> T, v: &[T]) -> &mut Self {
        for (x, v) in self.iter_mut().zip(v) {
            *x = op(*v)
        }
        self
    }

    fn translate(&mut self, c: T) -> &mut Self {
        //NB: translate is a scalar shift of all variables and is only
        //used only in the NN cone to force vectors into R^n_+
        self.scalarop(|x| x + c)
    }

    fn set(&mut self, c: T) -> &mut Self {
        self.scalarop(|_x| c)
    }

    fn scale(&mut self, c: T) -> &mut Self {
        self.scalarop(|x| x * c)
    }

    fn recip(&mut self) -> &mut Self {
        self.scalarop(T::recip)
    }

    fn sqrt(&mut self) -> &mut Self {
        self.scalarop(T::sqrt)
    }

    fn rsqrt(&mut self) -> &mut Self {
        self.scalarop(|x| T::recip(T::sqrt(x)))
    }

    fn negate(&mut self) -> &mut Self {
        self.scalarop(|x| -x)
    }

    fn hadamard(&mut self, y: &[T]) -> &mut Self {
        self.iter_mut().zip(y).for_each(|(x, y)| *x *= *y);
        self
    }

    fn clip(&mut self, min_thresh: T, max_thresh: T, min_new: T, max_new: T) -> &mut Self {
        self.scalarop(|x| x.clip(min_thresh, max_thresh, min_new, max_new))
    }

    fn normalize(&mut self) -> T {
        let norm = self.norm();
        if norm == T::zero() {
            return T::zero();
        }
        self.scale(norm.recip());
        norm
    }

    fn dot(&self, y: &[T]) -> T {
        self.iter()
            .zip(y)
            .fold(T::zero(), |acc, (&x, &y)| acc + x * y)
    }

    fn dot_shifted(z: &[T], s: &[T], dz: &[T], ds: &[T], α: T) -> T {
        assert_eq!(z.len(), s.len());
        assert_eq!(z.len(), dz.len());
        assert_eq!(s.len(), ds.len());

        let s_ds = s.iter().zip(ds.iter());
        let z_dz = z.iter().zip(dz.iter());
        let mut out = T::zero();
        for ((&s, &ds), (&z, &dz)) in s_ds.zip(z_dz) {
            let si = s + α * ds;
            let zi = z + α * dz;
            out += si * zi;
        }
        out
    }

    fn dist(&self, y: &Self) -> T {
        let dist2 = self
            .iter()
            .zip(y)
            .fold(T::zero(), |acc, (&x, &y)| acc + T::powi(x - y, 2));
        T::sqrt(dist2)
    }

    fn sumsq(&self) -> T {
        self.dot(self)
    }

    // 2-norm
    fn norm(&self) -> T {
        T::sqrt(self.sumsq())
    }

    //scaled norm of elementwise produce self.*v
    fn norm_scaled(&self, v: &[T]) -> T {
        assert_eq!(self.len(), v.len());
        let total = self.iter().zip(v).fold(T::zero(), |acc, (&x, &y)| {
            let prod = x * y;
            acc + prod * prod
        });
        T::sqrt(total)
    }

    // Returns infinity norm, ignoring NaNs
    fn norm_inf(&self) -> T {
        let mut out = T::zero();
        for v in self.iter().map(|v| v.abs()) {
            out = if v > out { v } else { out };
        }
        out
    }

    // Returns one norm
    fn norm_one(&self) -> T {
        self.iter().fold(T::zero(), |acc, v| acc + v.abs())
    }

    fn minimum(&self) -> T {
        self.iter().fold(T::infinity(), |r, &s| T::min(r, s))
    }

    fn maximum(&self) -> T {
        self.iter().fold(-T::infinity(), |r, &s| T::max(r, s))
    }

    fn mean(&self) -> T {
        let mean = if self.is_empty() {
            T::zero()
        } else {
            let num = self.iter().fold(T::zero(), |r, &s| r + s);
            let den = T::from_usize(self.len()).unwrap();
            num / den
        };
        mean
    }

    fn is_finite(&self) -> bool {
        self.iter().all(|&x| T::is_finite(x))
    }

    fn axpby(&mut self, a: T, x: &[T], b: T) -> &mut Self {
        assert_eq!(self.len(), x.len());

        let yx = self.iter_mut().zip(x);
        yx.for_each(|(y, x)| *y = a * (*x) + b * (*y));
        self
    }

    fn waxpby(&mut self, a: T, x: &[T], b: T, y: &[T]) -> &mut Self {
        assert_eq!(self.len(), x.len());
        assert_eq!(self.len(), y.len());

        let xy = x.iter().zip(y);

        for (w, (x, y)) in self.iter_mut().zip(xy) {
            *w = a * (*x) + b * (*y);
        }
        self
    }
}

impl<T: FloatT> MatrixMath<T, [T]> for CscMatrix<T> {
    //scalar mut operations
    fn scale(&mut self, c: T) {
        self.nzval.scale(c);
    }

    fn negate(&mut self) {
        self.nzval.negate();
    }

    fn col_norms(&self, norms: &mut [T]) {
        norms.fill(T::zero());
        self.col_norms_no_reset(norms);
    }

    fn col_norms_no_reset(&self, norms: &mut [T]) {
        assert_eq!(norms.len(), self.colptr.len() - 1);

        // for (i,v) in norms.iter_mut().enumerate(){
        //     for j in self.colptr[i]..self.colptr[i + 1]{
        //         let tmp = T::abs(self.nzval[j]);
        //         *v = T::max(*v,tmp);
        //     }
        // }
        for (i, v) in norms.iter_mut().enumerate() {
            *v = self
                .nzval
                .iter()
                .take(self.colptr[i + 1])
                .skip(self.colptr[i])
                .fold(*v, |m, &nzval| T::max(m, T::abs(nzval)));
        }
    }

    fn col_norms_sym(&self, norms: &mut [T]) {
        norms.fill(T::zero());
        self.col_norms_sym_no_reset(norms);
    }

    fn col_norms_sym_no_reset(&self, norms: &mut [T]) {
        assert_eq!(norms.len(), self.colptr.len() - 1);

        for i in 0..norms.len() {
            for j in self.colptr[i]..self.colptr[i + 1] {
                let tmp = T::abs(self.nzval[j]);
                let r = self.rowval[j];
                norms[i] = T::max(norms[i], tmp);
                norms[r] = T::max(norms[r], tmp);
            }
        }
    }

    fn row_norms(&self, norms: &mut [T]) {
        norms.fill(T::zero());
        self.row_norms_no_reset(norms);
    }

    fn row_norms_no_reset(&self, norms: &mut [T]) {
        assert_eq!(self.rowval.len(), *self.colptr.last().unwrap());

        for (row, val) in self.rowval.iter().zip(self.nzval.iter()) {
            norms[*row] = T::max(norms[*row], T::abs(*val));
        }
    }

    fn lscale(&mut self, l: &[T]) {
        let rows = &self.rowval;
        let vals = &mut self.nzval;

        for (val, row) in vals.iter_mut().zip(rows) {
            *val *= l[*row];
        }
    }

    fn rscale(&mut self, r: &[T]) {
        let colptr = &self.colptr;
        let vals = &mut self.nzval;

        assert_eq!(vals.len(), *colptr.last().unwrap());
        for i in 0..self.n {
            vals[colptr[i]..colptr[i + 1]].scale(r[i]);
        }
    }

    fn lrscale(&mut self, l: &[T], r: &[T]) {
        assert_eq!(self.nzval.len(), *self.colptr.last().unwrap());

        for (col, &ri) in r.iter().enumerate() {
            let (first, last) = (self.colptr[col], self.colptr[col + 1]);
            let vals = &mut self.nzval[first..last];
            let rows = &self.rowval[first..last];

            for (val, row) in vals.iter_mut().zip(rows) {
                *val *= l[*row] * ri;
            }
        }
    }

    fn gemv(&self, y: &mut [T], trans: MatrixShape, x: &[T], a: T, b: T) {
        match trans {
            MatrixShape::N => _csc_axpby_N(self, y, x, a, b),
            MatrixShape::T => _csc_axpby_T(self, y, x, a, b),
        }
    }

    fn symv(&self, y: &mut [T], tri: MatrixTriangle, x: &[T], a: T, b: T) {
        //NB: the triangle argument doesn't actually do
        //anything here, and the call is the same either
        //way.  The argument serves only as a reminder that
        //the caller should only pass a triangular form
        match tri {
            MatrixTriangle::Triu => _csc_symv_unsafe(self, y, x, a, b),
            MatrixTriangle::Tril => _csc_symv_unsafe(self, y, x, a, b),
        }
    }

    fn quad_form(&self, y: &[T], x: &[T]) -> T {
        _csc_quad_form(self, y, x)
    }
}

#[allow(non_snake_case)]
fn _csc_symv_safe<T: FloatT>(A: &CscMatrix<T>, y: &mut [T], x: &[T], a: T, b: T) {
    y.scale(b);

    assert!(x.len() == A.n);
    assert!(y.len() == A.n);
    assert!(A.n == A.m);

    for (col, &xcol) in x.iter().enumerate() {
        let first = A.colptr[col];
        let last = A.colptr[col + 1];
        let rows = &A.rowval[first..last];
        let nzvals = &A.nzval[first..last];

        for (&row, &Aij) in rows.iter().zip(nzvals.iter()) {
            y[row] += a * Aij * xcol;

            if row != col {
                //don't double up on the diagonal
                y[col] += a * Aij * x[row];
            }
        }
    }
}

// Safety: The function below checks that x and y are compatible with
// the dimensions of A, so safety is assured so long as the the matrix
// A as rowval and colptr arrays that are consistent with its dimension.
// A bounds checked version is provided above.
//
// This `unsafe`d version is preferred the multiplication K*x, with K
// the symmetric KKT matrix, is used heavily in iterative refinement of
// direct linear solves.
#[allow(non_snake_case)]
fn _csc_symv_unsafe<T: FloatT>(A: &CscMatrix<T>, y: &mut [T], x: &[T], a: T, b: T) {
    y.scale(b);

    assert!(x.len() == A.n);
    assert!(y.len() == A.n);
    assert!(A.n == A.m);
    unsafe {
        for (col, &xcol) in x.iter().enumerate() {
            let first = *A.colptr.get_unchecked(col);
            let last = *A.colptr.get_unchecked(col + 1);

            for j in first..last {
                let row = *A.rowval.get_unchecked(j);
                let Aij = *A.nzval.get_unchecked(j);
                *y.get_unchecked_mut(row) += a * Aij * xcol;

                if row != col {
                    //don't double up on the diagonal
                    *y.get_unchecked_mut(col) += a * Aij * (*x.get_unchecked(row));
                }
            }
        }
    }
}

#[allow(non_snake_case)]
#[allow(clippy::comparison_chain)]
fn _csc_quad_form<T: FloatT>(M: &CscMatrix<T>, y: &[T], x: &[T]) -> T {
    assert_eq!(M.n, M.m);
    assert_eq!(x.len(), M.n);
    assert_eq!(y.len(), M.n);
    assert!(M.colptr.len() == M.n + 1);
    assert!(M.nzval.len() == M.rowval.len());

    if M.n == 0 {
        return T::zero();
    }

    let mut out = T::zero();

    for col in 0..M.n {
        //column number

        let mut tmp1 = T::zero();
        let mut tmp2 = T::zero();

        //start / stop indices for the current column
        let first = M.colptr[col];
        let last = M.colptr[col + 1];

        let values = &M.nzval[first..last];
        let rows = &M.rowval[first..last];
        let iter = values.iter().zip(rows.iter());

        for (&Mv, &row) in iter {
            if row < col {
                //triu terms only
                tmp1 += Mv * x[row];
                tmp2 += Mv * y[row];
            } else if row == col {
                out += Mv * x[col] * y[col];
            } else {
                panic!("Input matrix should be triu form.");
            }
        }
        out += tmp1 * y[col] + tmp2 * x[col]
    }
    out
}

// sparse matrix-vector multiply, no transpose
#[allow(non_snake_case)]
fn _csc_axpby_N<T: FloatT>(A: &CscMatrix<T>, y: &mut [T], x: &[T], a: T, b: T) {
    //first do the b*y part
    if b == T::zero() {
        y.fill(T::zero())
    } else if b == T::one() {
    } else if b == -T::one() {
        y.negate();
    } else {
        y.scale(b);
    }

    // if a is zero, we're done
    if a == T::zero() {
        return;
    }

    assert_eq!(A.nzval.len(), *A.colptr.last().unwrap());
    assert_eq!(x.len(), A.n);

    //y += A*x
    if a == T::one() {
        for (j, xj) in x.iter().enumerate().take(A.n) {
            for i in A.colptr[j]..A.colptr[j + 1] {
                y[A.rowval[i]] += A.nzval[i] * *xj;
            }
        }
    } else if a == -T::one() {
        for (j, xj) in x.iter().enumerate().take(A.n) {
            for i in A.colptr[j]..A.colptr[j + 1] {
                y[A.rowval[i]] -= A.nzval[i] * *xj;
            }
        }
    } else {
        for (j, xj) in x.iter().enumerate().take(A.n) {
            for i in A.colptr[j]..A.colptr[j + 1] {
                y[A.rowval[i]] += a * A.nzval[i] * *xj;
            }
        }
    }
}

// sparse matrix-vector multiply, transposed
#[allow(non_snake_case)]
fn _csc_axpby_T<T: FloatT>(A: &CscMatrix<T>, y: &mut [T], x: &[T], a: T, b: T) {
    //first do the b*y part
    if b == T::zero() {
        y.fill(T::zero())
    } else if b == T::one() {
    } else if b == -T::one() {
        y.negate();
    } else {
        y.scale(b);
    }

    // if a is zero, we're done
    if a == T::zero() {
        return;
    }

    assert_eq!(A.nzval.len(), *A.colptr.last().unwrap());
    assert_eq!(x.len(), A.m);

    //y += A*x
    if a == T::one() {
        for (j, yj) in y.iter_mut().enumerate().take(A.n) {
            for k in A.colptr[j]..A.colptr[j + 1] {
                *yj += A.nzval[k] * x[A.rowval[k]];
            }
        }
    } else if a == -T::one() {
        for (j, yj) in y.iter_mut().enumerate().take(A.n) {
            for k in A.colptr[j]..A.colptr[j + 1] {
                *yj -= A.nzval[k] * x[A.rowval[k]];
            }
        }
    } else {
        for (j, yj) in y.iter_mut().enumerate().take(A.n) {
            for k in A.colptr[j]..A.colptr[j + 1] {
                *yj += a * A.nzval[k] * x[A.rowval[k]];
            }
        }
    }
}
