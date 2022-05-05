use super::*;
pub mod cscmatrix;
pub use cscmatrix::*;


impl<T> ScalarMath<T> for T
where
T: FloatT,
{
    fn clip(s: T, min_thresh: T, max_thresh: T, min_new: T, max_new: T) -> T
    {
        if s < min_thresh {min_new}
        else if s > max_thresh {max_new}
        else {s}
    }
}

impl<T> VectorMath<T> for [T]
where
    T: FloatT,
{

    fn copy_from(&mut self, src: &[T]){
        self.copy_from_slice(src);
    }

    fn scalarop(&mut self, op: impl Fn(T) -> T) {
        for x in self {*x = op(*x)}
    }

    fn scalarop_from(&mut self, op: impl Fn(T) -> T, v: &[T]) {
        for (x,v) in self.iter_mut().zip(v) {*x = op(*v)}
    }

    fn translate(&mut self, c: T) {
        self.scalarop(|x| x+c);
    }

    fn scale(&mut self, c: T) {
        self.scalarop(|x| x*c);
    }

    fn reciprocal(&mut self) {
        self.scalarop(T::recip);
    }

    fn sqrt(&mut self) {
        self.scalarop(T::sqrt);
    }

    fn rsqrt(&mut self) {
        self.scalarop(|x| T::recip(T::sqrt(x)));
    }

    fn negate(&mut self) {
        self.scalarop(|x| -x);
    }

    fn hadamard(&mut self, y: &[T]) {
        self.iter_mut().zip(y).for_each(|(x,y)| *x *= *y);
    }

    fn clip(& mut self, min_thresh: T, max_thresh: T, min_new: T, max_new: T){
        self.scalarop(|x| T::clip(x, min_thresh, max_thresh, min_new, max_new) );
    }

    fn dot(&self, y: &[T]) -> T {
        self.iter().zip(y).fold(T::zero(),|acc, (&x, &y)| acc + x * y)
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

        assert_eq!(self.len(),v.len());
        let total = self.iter().zip(v).
        fold(T::zero(),|acc, (&x, &y)|{
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
        self.iter().fold(T::zero(), |acc,v| acc + v.abs())
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
            let den = T::from(self.len()).unwrap();
            num / den
        };
        mean
    }

    fn axpby(&mut self, a: T, x: &[T], b: T) {
        assert_eq!(self.len(), x.len());

        //handle b = 1 / 0 / -1 separately
        let yx = self.iter_mut().zip(x);
        if b == T::zero() {
            yx.for_each(|(y, x)| *y = a * (*x));
        } else if b == T::one() {
            yx.for_each(|(y, x)| *y = a * (*x) + (*y));
        } else if b == -T::one() {
            yx.for_each(|(y, x)| *y = a * (*x) - (*y));
        } else {
            yx.for_each(|(y, x)| *y = a * (*x) + b * (*y));
        }
    }

    fn waxpby(&mut self, a: T, x: &[T], b: T, y: &[T]) {
        assert_eq!(self.len(), x.len());
        assert_eq!(self.len(), y.len());

        let xy = x.iter().zip(y);

        for (w, (x, y)) in self.iter_mut().zip(xy) {
            *w = a * (*x) + b * (*y);
        }
    }
}



impl<T: FloatT> MatrixMath<T,[T]> for CscMatrix<T>
where
    T: FloatT
{

    //matrix properties
    fn nrows(&self) -> usize {self.m}
    fn ncols(&self) -> usize {self.n}
    fn nnz(&self) -> usize {self.colptr[self.n]}
    fn is_square(&self) -> bool {self.m == self.n}

    //scalar mut operations
    fn scale(&mut self, c: T){
        self.nzval.scale(c);
    }

    fn col_norms(&self, norms: &mut [T]){
        norms.fill(T::zero());
        self.col_norms_no_reset(norms);
    }

    fn col_norms_no_reset(&self, norms: &mut [T]){

        assert_eq!(norms.len(),self.colptr.len()-1);

        for (i,v) in norms.iter_mut().enumerate(){
            for j in self.colptr[i]..self.colptr[i + 1]{
                let tmp = T::abs(self.nzval[j]);
                *v = T::max(*v,tmp);
            }
        }
    }

    fn col_norms_sym(&self, norms: &mut [T]){
        norms.fill(T::zero());
        self.col_norms_sym_no_reset(norms);
    }

    fn col_norms_sym_no_reset(&self, norms: &mut [T]){

        assert_eq!(norms.len(),self.colptr.len()-1);

        for i in 0 .. norms.len(){
            for j in self.colptr[i]..self.colptr[i + 1]{
                let tmp = T::abs(self.nzval[j]);
                let r   = self.rowval[j];
                norms[i] = T::max(norms[i],tmp);
                norms[r] = T::max(norms[r],tmp);
            }
        }
    }

    fn row_norms(&self, norms: &mut [T]){
        norms.fill(T::zero());
        self.row_norms_no_reset(norms);
    }

    fn row_norms_no_reset(&self, norms: &mut [T]){

        assert_eq!(self.rowval.len(),*self.colptr.last().unwrap());

        for (row,val) in self.rowval.iter().zip(self.nzval.iter()){
            norms[*row] = T::max(norms[*row],T::abs(*val));
        }
    }

    fn lscale(&mut self, l: &[T]){

        let rows = &self.rowval;
        let vals = &mut self.nzval;

        for (val,row) in vals.iter_mut().zip(rows) {
            *val *= l[*row];
        }
    }

    fn rscale(&mut self, r: &[T]){

        let colptr = &self.colptr;
        let vals = &mut self.nzval;

        assert_eq!(vals.len(),*colptr.last().unwrap());
        for i in 0..self.n {
            vals[colptr[i]..colptr[i+1]].scale(r[i]);
        }

    }

    fn lrscale(&mut self, l: &[T], r: &[T]){

        assert_eq!(self.nzval.len(),*self.colptr.last().unwrap());

        for (col,&ri) in r.iter().enumerate() {
            let (first,last) = (self.colptr[col],self.colptr[col+1]);
            let  vals = &mut self.nzval[first..last];
            let rows = &self.rowval[first..last];

            for (val,row) in vals.iter_mut().zip(rows){
                *val *= l[*row] * ri;
            }
        }
    }

    fn gemv(&self, y: &mut [T], trans: MatrixShape, x: &[T], a:T, b:T){

        match trans {
            MatrixShape::N => _csc_axpby_N(self, y, x, a, b),
            MatrixShape::T => _csc_axpby_T(self, y, x, a, b),
        }

    }

    fn symv(&self, y: &mut [T], tri: MatrixTriangle, x: &[T], a:T, b:T){

        //NB: the triangle argument doesn't actually do
        //anything here, and the call is the same either
        //way.  The argument serves only as a reminder that
        //the caller should only pass a triangular form
        match tri {
            MatrixTriangle::Triu => _csc_symv(self, y, x, a, b),
            MatrixTriangle::Tril => _csc_symv(self, y, x, a, b),
        }
    }

    fn symdot(&self, y : &[T], x : &[T]) -> T {
        _csc_quad_form(self,y,x)
    }
}

#[allow(non_snake_case)]
fn _csc_symv<T: FloatT>(A: &CscMatrix<T>, y: &mut [T], x: &[T], a:T, b:T){

    y.scale(b);

    //PJG: some iterator magic needed here

    for col in 0..A.n {
        for i in A.colptr[col]..A.colptr[col+1] {
            let row   = A.rowval[i];
            let Aij   = A.nzval[i];
            y[row] += a * Aij * x[col];
            if row != col {
                //don't double up on the diagonal
                y[col] += a * Aij * x[row];
            }
        }
    }

}

#[allow(non_snake_case)]
#[allow(clippy::comparison_chain)]
fn _csc_quad_form<T: FloatT>(M: &CscMatrix<T>, y: &[T], x: &[T]) -> T{

    let (m, n) = (M.m, M.n);

    assert_eq!(x.len(), m);
    assert_eq!(y.len(), n);

    if m == 0 || n == 0{
        return T::zero()
    }

    let Mc = &M.colptr;
    let Mr = &M.rowval;
    let Mv = &M.nzval;

    let mut out = T::zero();

    for j in 0..n {   //col number
        let mut tmp1 = T::zero();
        let mut tmp2 = T::zero();
        for p in Mc[j]..Mc[j+1] {
            let i = Mr[p];   //row number
            if i < j {
                //triu terms only
                tmp1 += Mv[p]*x[i];
                tmp2 += Mv[p]*y[i];
            }
            else if i == j {
                out += Mv[p]*x[i]*y[i];
            }
            else{
                panic!("Input matrix should be triu form.");
            }   
        }
        out += tmp1*y[j] + tmp2*x[j]
    }
    out

}


// sparse matrix-vector multiply, no transpose
#[allow(non_snake_case)]
fn _csc_axpby_N<T: FloatT>(A: &CscMatrix<T>, y: &mut [T], x: &[T], a:T, b:T)
where
    T:FloatT,
{

  //first do the b*y part
  if b == T::zero() {y.fill(T::zero())}
  else if b == T::one() {}
  else if b == -T::one() {y.negate()}
  else {y.scale(b)}

  // if a is zero, we're done
  if a == T::zero() {return}

  assert_eq!(A.nzval.len(),*A.colptr.last().unwrap());
  assert_eq!(x.len(),A.n);

  //y += A*x
  if a == T::one(){
      for (j, xj) in x.iter().enumerate().take(A.n) {
          for i in A.colptr[j]..A.colptr[j+1]{
              y[A.rowval[i]] += A.nzval[i] * *xj;
          }
      }
  }
  else if a == -T::one(){
      for (j, xj) in x.iter().enumerate().take(A.n) {
          for i in A.colptr[j]..A.colptr[j+1]{
              y[A.rowval[i]] -= A.nzval[i] * *xj;
          }
      }
  }
  else{
      for (j,xj) in x.iter().enumerate().take(A.n) {
          for i in A.colptr[j]..A.colptr[j+1]{
              y[A.rowval[i]] += a * A.nzval[i] * *xj;
          }
      }
  }

}

// sparse matrix-vector multiply, transposed
#[allow(non_snake_case)]
fn _csc_axpby_T<T: FloatT>(A: &CscMatrix<T>, y: &mut [T], x: &[T], a:T, b:T){

    //first do the b*y part
    if b == T::zero() {y.fill(T::zero())}
    else if b == T::one() {}
    else if b == -T::one() {y.negate()}
    else {y.scale(b)}

    // if a is zero, we're done
    if a == T::zero() {return}

    assert_eq!(A.nzval.len(),*A.colptr.last().unwrap());
    assert_eq!(x.len(),A.m);

    //y += A*x
    if a == T::one(){
        for (j,yj) in y.iter_mut().enumerate().take(A.n) {
            for k in A.colptr[j]..A.colptr[j+1]{
                *yj += A.nzval[k] * x[A.rowval[k]];
            }
        }
    }
    else if a == -T::one(){
        for (j,yj) in y.iter_mut().enumerate().take(A.n) {
            for k in A.colptr[j]..A.colptr[j+1]{
                *yj -= A.nzval[k] * x[A.rowval[k]];
            }
        }
    }
    else{
        for (j,yj) in y.iter_mut().enumerate().take(A.n) {
            for k in A.colptr[j]..A.colptr[j+1]{
                *yj += a * A.nzval[k] * x[A.rowval[k]];
            }
        }
    }

}
