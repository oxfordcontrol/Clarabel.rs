use super::{cones::Cone, CoreSettings};
use clarabel_algebra::*;
use clarabel_timers::*;

pub trait ProblemData<T: FloatT> {
    type V: Variables<T>;
    type C: Cone<T>;
    type SE: Settings<T>;

    fn equilibrate(&mut self, cones: &Self::C, settings: &Self::SE);
}

pub trait Variables<T: FloatT> {
    type D: ProblemData<T>;
    type R: Residuals<T>;
    type C: Cone<T>;

    fn calc_mu(&mut self, residuals: &Self::R, cones: &Self::C) -> T;

    fn calc_affine_step_rhs(&mut self, residuals: &Self::R, variables: &Self, cones: &Self::C);

    #[allow(clippy::too_many_arguments)]
    fn calc_combined_step_rhs(
        &mut self,
        residuals: &Self::R,
        variables: &Self,
        cones: &Self::C,
        step: &mut Self, //mut allows step to double as working space
        σ: T,
        μ: T,
    );

    fn calc_step_length(&mut self, step_lhs: &Self, cones: &Self::C) -> T;
    fn add_step(&mut self, step_lhs: &Self, α: T);
    fn shift_to_cone(&mut self, cones: &Self::C);
    fn scale_cones(&self, cones: &mut Self::C);
}

pub trait Residuals<T: FloatT> {
    type D: ProblemData<T>;
    type V: Variables<T>;

    fn update(&mut self, variables: &Self::V, data: &Self::D);
}

pub trait KKTSystem<T: FloatT> {
    type D: ProblemData<T>; 
    type V: Variables<T>; 
    type C: Cone<T>;
    type SE: Settings<T>;

    fn update(&mut self, data: &Self::D, cones: &Self::C, settings: &Self::SE);

    #[allow(clippy::too_many_arguments)]
    fn solve(
        &mut self,
        step_lhs: &mut Self::V,
        step_rhs: &Self::V,
        data: &Self::D,
        variables: &Self::V,
        cones: &Self::C,
        steptype: &'static str,
        settings: &Self::SE,
    );

    fn solve_initial_point(&mut self, variables: &mut Self::V, data: &Self::D, settings: &Self::SE);
}

pub trait InfoPrint<T> 
where 
    T:FloatT
{
    type D: ProblemData<T>;
    type C: Cone<T>;
    type SE: Settings<T>;

    fn print_configuration(&self, settings: &Self::SE, data: &Self::D, cones: &Self::C);
    fn print_status_header(&self, settings: &Self::SE);
    fn print_status(&self, settings: &Self::SE);
    fn print_footer(&self, settings: &Self::SE);
}

pub trait Info<T> : InfoPrint<T>
where 
    T: FloatT
{
    type V: Variables<T>;
    type R: Residuals<T>;

    fn reset(&mut self, timers: &mut Timers);
    fn finalize(&mut self, timers: &mut Timers);
    fn update(&mut self, data: &Self::D, variables: &Self::V, residuals: &Self::R, timers: &Timers);
    fn check_termination(&mut self, residuals: &Self::R, settings: &Self::SE) -> bool;
    fn save_scalars(&mut self, μ: T, α: T, σ: T, iter: u32);
}

pub trait Solution<T: FloatT> {
    type D: ProblemData<T>;
    type V: Variables<T>;
    type I: Info<T>;

    fn finalize(&mut self, data: &Self::D, variables: &Self::V, info: &Self::I);
}

pub trait Settings<T: FloatT> {
    fn core(&self) -> &CoreSettings<T>;
    fn core_mut(&mut self) -> &mut CoreSettings<T>;
}
