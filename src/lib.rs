//Rust hates greek characters
#![allow(confusable_idents)]

pub mod algebra;
pub mod qdldl;
pub mod solver;
pub mod timers;

#[cfg(feature = "python")]
pub mod python;
