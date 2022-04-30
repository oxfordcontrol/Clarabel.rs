use crate::algebra::*;
use crate::default::*;

pub struct DefaultSolveInfo<T> {
    foo: T,
}


impl<T:FloatT> DefaultSolveInfo<T>
{
    pub fn new()->Self{
        Self{foo:T::zero()}
    }
}

impl<T: FloatT> SolveInfo<T> for DefaultSolveInfo<T>
{
    type D = DefaultProblemData<T>;
    type V = DefaultVariables<T>;
    type R = DefaultResiduals<T>;
    type C = ConeSet<T>;

    fn reset(&mut self){todo!();}
    fn finalize(&mut self){todo!();}

    fn print_header(
        &mut self,
        settings: &Settings<T>,
        data: &DefaultProblemData<T>,
        cones: &ConeSet<T>)
    {
        todo!();    
    }

    fn print_status(&mut self, settings: &Settings<T>){todo!();}
    fn print_footer(&mut self, settings: &Settings<T>){todo!();}

    fn update(
        &mut self,
        data: &DefaultProblemData<T>,
        variables: &DefaultVariables<T>,
        residuals: &DefaultResiduals<T>,
        settings: &Settings<T>,
    )
    {
        todo!();
    }

    fn check_termination(
        &mut self,
        residuals: &DefaultResiduals<T>,
        settings: &Settings<T>) -> bool
    {
        todo!();
        true
    }

    fn save_scalars(&mut self, μ: T, α: T, σ: T, iter: i32)
    {
        todo!();
    }

}
