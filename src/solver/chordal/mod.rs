// Julia version of chordal decomposition that was
// ported from COSMO has a lot of explicitly index loops.
// Allow in this module for the sake of easier porting
#![allow(clippy::needless_range_loop)]

type VertexSet = indexmap::IndexSet<usize>;

mod chordal_info;
mod decomp;
mod merge;
mod sparsity_pattern;
mod supernode_tree;

pub(crate) use chordal_info::*;
pub(crate) use merge::*;
pub(crate) use sparsity_pattern::*;
pub(crate) use supernode_tree::*;
