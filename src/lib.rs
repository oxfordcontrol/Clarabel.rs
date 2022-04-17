//PJG: Temporary allows during development
#![allow(confusable_idents)]
#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_mut)]

//PJG: Are both use and mod required throughout?
//Need a general cleanup of these header like
//declarations

pub mod algebra;
pub mod cones;
pub mod conicvector;
pub mod tests;
pub mod default;
pub mod settings;
pub use cones::*;
pub use algebra::*;
pub use settings::Settings;


pub trait ProblemData<T: FloatT> {

    fn equilibrate(&mut self, cones: &ConeSet<T>, settings: &Settings);
}
