use crate::algebra::*;
use crate::default::*;

pub struct DefaultSolveResult<T> {
    foo: T,
}

impl<T:FloatT> DefaultSolveResult<T>
{
    pub fn new(m: usize, n: usize)->Self{
        todo!();
        Self{foo:T::zero()}
    }
}


impl<T:FloatT> SolveResult<T> for DefaultSolveResult<T> {

    type D = DefaultProblemData<T>;
    type V = DefaultVariables<T>;
    type SI = DefaultSolveInfo<T>;

    fn finalize(
        &mut self,
        data: &DefaultProblemData<T>,
        variables: &DefaultVariables<T>,
        info: &DefaultSolveInfo<T>)
    {
        todo!();
    }
}
