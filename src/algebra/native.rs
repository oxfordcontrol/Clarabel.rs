use super::*;

impl<T> VectorMathOps<T> for [T]
where
    T: FloatT,
{

    fn translate(&mut self, c: T) {
        self.iter_mut().for_each(|x| *x += c);
    }

    fn scale(&mut self, c: T) {
        self.iter_mut().for_each(|x| *x *= c);
    }

    fn reciprocal(&mut self) {
        self.iter_mut().for_each(|x| *x = T::recip(*x));
    }

    fn dot(&self, y: &[T]) -> T
    {
        self.iter().zip(y).map(|(&x, &y)| x * y).sum()
    }

    fn sumsq(&self) -> T
    {
        self.dot(self)
    }

    // 2-norm
    fn norm(&self) -> T {
        T::sqrt(self.sumsq())
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
        self.iter().map(|v| v.abs()).sum::<T>()
    }

    fn minimum(&self) -> T
    {
        self.iter().fold(T::infinity(), |r, &s| T::min(r,s))
    }

    fn maximum(&self) -> T
    {
        self.iter().fold(T::infinity(), |r, &s| T::max(r,s))
    }

    fn mean(&self) -> T
    {
        if self.len() == 0 {
            return T::zero()
        }

        else{
            let num = self.iter().fold(T::zero(), |r, &s| r + s);
            let den = T::from(self.len()).unwrap();
            return num / den
        }
    }

    fn axpby(&mut self, a: T, x: &[T], b :T){

        assert_eq!(self.len(),x.len());

        //handle b = 1 / 0 / -1 separately
        if b == T::zero() {
            self.iter_mut().zip(x.iter()).for_each(|(y,x)| *y = a*(*x));
        }
        else if b == T::one() {
            self.iter_mut().zip(x.iter()).for_each(|(y,x)| *y = a*(*x)+(*y));
        }
        else if b == -T::one() {
            self.iter_mut().zip(x.iter()).for_each(|(y,x)| *y = a*(*x)-(*y));
        }
        else {
            self.iter_mut().zip(x.iter()).for_each(|(y,x)| *y = a*(*x)+b*(*y));
        }
    }

    fn waxpby(&mut self, a: T, x: &[T], b :T, y: &[T]){
        assert_eq!(self.len(),x.len());
        assert_eq!(self.len(),y.len());

        let xy = x.iter().zip(y.iter());

        for (w, (x,y)) in self.iter_mut().zip(xy) {
            *w = a*(*x) * b*(*y);
        }
    }

}
