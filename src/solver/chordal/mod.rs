type VertexSet = indexmap::IndexSet<usize>;

mod chordal_info;
mod decomp;
mod merge;
mod sparsity_pattern;
mod supernode_tree;

pub(crate) use chordal_info::*;
pub(crate) use decomp::*;
pub(crate) use merge::*;
pub(crate) use sparsity_pattern::*;
pub(crate) use supernode_tree::*;
