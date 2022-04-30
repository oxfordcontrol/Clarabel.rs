use crate::algebra::*;
use crate::default::*;

pub struct DefaultKKTSystem<T> {
    foo: T,
}

impl<T:FloatT> DefaultKKTSystem<T>
{
    pub fn new(cones: &ConeSet<T>, settings: &Settings<T>)->Self{
        todo!();
        Self{foo:T::zero()}
    }
}


impl<T> KKTSystem<T> for DefaultKKTSystem<T>
where
    T: FloatT,
{

    type D = DefaultProblemData<T>;
    type V = DefaultVariables<T>;
    type C = ConeSet<T>;

    fn update(&mut self, data: &DefaultProblemData<T>, cones: &ConeSet<T>)
    {
        todo!();
    }

    fn solve(
        &mut self,
        step_lhs: &DefaultVariables<T>,
        step_rhs: &DefaultVariables<T>,
        data: &DefaultProblemData<T>,
        variables: &DefaultVariables<T>,
        cones: &ConeSet<T>,
        steptype: &str,
    )
    {
        todo!();
    }

    fn solve_initial_point(
        &mut self,
        variables: &DefaultVariables<T>,
        data: &DefaultProblemData<T>)
    {
        todo!();
    }
}
