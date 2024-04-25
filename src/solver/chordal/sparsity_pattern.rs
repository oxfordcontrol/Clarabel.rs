#![allow(non_snake_case)]
use crate::algebra::*;
use crate::solver::chordal::*;

// ---------------------------
// Struct to hold clique and sparsity data for a constraint
// ---------------------------

#[derive(Debug)]
pub(crate) struct SparsityPattern {
    pub(crate) sntree: SuperNodeTree,
    pub(crate) ordering: Vec<usize>,
    pub(crate) orig_index: usize, // original index of the cone being decomposed
}

impl SparsityPattern {
    // constructor for sparsity pattern
    pub(crate) fn new<T: FloatT>(
        L: CscMatrix<T>,
        mut ordering: Vec<usize>,
        orig_index: usize,
        merge_method: &str,
    ) -> Self {
        let mut sntree = SuperNodeTree::new(&L);

        // clique merging only if more than one clique present

        if sntree.n_cliques > 1 {
            match merge_method {
                "none" => {
                    NoMergeStrategy::new().merge_cliques(&mut sntree);
                }
                "parent_child" => {
                    ParentChildMergeStrategy::new().merge_cliques(&mut sntree);
                }
                "clique_graph" => {
                    CliqueGraphMergeStrategy::new().merge_cliques(&mut sntree);
                }
                _ => {
                    panic! {"Unrecognized merge strategy"};
                }
            }
        }

        // reorder vertices in supernodes to have consecutive order
        // necessary for equal column structure for psd completion
        sntree.reorder_snode_consecutively(&mut ordering);

        // for each clique determine the number of entries of the block
        // represented by that clique
        sntree.calculate_block_dimensions();

        Self {
            sntree,
            ordering,
            orig_index,
        }
    }
}
