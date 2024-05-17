use super::*;
use crate::solver::chordal::*;

pub(crate) struct ParentChildMergeStrategy {
    stop: bool,
    clique_index: usize,
    t_fill: usize,
    t_size: usize,
}

impl ParentChildMergeStrategy {
    pub(crate) fn new() -> Self {
        let t_fill = 8; // PJG: make settable
        let t_size = 8;
        Self {
            stop: false,
            clique_index: 0,
            t_fill,
            t_size,
        }
    }
}

impl MergeStrategy for ParentChildMergeStrategy {
    fn initialise(&mut self, t: &mut SuperNodeTree) {
        // start with node that has second highest order
        self.clique_index = t.snode.len() - 2;
    }

    fn is_done(&self) -> bool {
        self.stop
    }

    // Traverse tree `t` in descending topological order and return parent and clique (root has highest order).
    fn traverse(&mut self, t: &SuperNodeTree) -> Option<(usize, usize)> {
        let c = t.snode_post[self.clique_index];
        Some((t.snode_parent[c], c))
    }

    fn evaluate(&mut self, t: &SuperNodeTree, cand: (usize, usize)) -> bool {
        if self.stop {
            return false;
        }

        let (parent, child) = cand;

        let (dim_parent_snode, dim_parent_sep) = clique_dim(t, parent);
        let (dim_clique_snode, dim_clique_sep) = clique_dim(t, child);

        let fill = fill_in(
            dim_clique_snode,
            dim_clique_sep,
            dim_parent_snode,
            dim_parent_sep,
        );
        let max_snode = std::cmp::max(dim_clique_snode, dim_parent_snode);

        fill <= self.t_fill || max_snode <= self.t_size
    }

    fn merge_two_cliques(&self, t: &mut SuperNodeTree, cand: (usize, usize)) {
        // determine which clique is the parent
        let (p, ch) = determine_parent(t, cand.0, cand.1);

        // merge child's vertex sets into parent's vertex set
        set_union_into_indexed(&mut t.snode, p, ch);
        t.snode[ch].clear();
        t.separators[ch].clear();

        // update parent structure
        for &grandch in t.snode_children[ch].iter() {
            t.snode_parent[grandch] = p;
        }
        t.snode_parent[ch] = INACTIVE_NODE;

        // update children structure
        t.snode_children[p].shift_remove(&ch); //preserves order for consistency with Julia
        set_union_into_indexed(&mut t.snode_children, p, ch);
        t.snode_children[ch].clear();

        // decrement number of cliques in tree
        t.n_cliques -= 1;
    }

    fn update_strategy(&mut self, _t: &SuperNodeTree, _cand: (usize, usize), _do_merge: bool) {
        // try to merge last node of order 0, then stop
        if self.clique_index == 0 {
            self.stop = true
        }
        // otherwise decrement node index
        else {
            self.clique_index -= 1
        }
    }

    fn post_process_merge(&mut self, t: &mut SuperNodeTree) {
        // the merging creates empty supernodes and seperators, recalculate a
        // post order for the supernodes (shrinks to length t.n_cliques )
        post_order(
            &mut t.snode_post,
            &t.snode_parent,
            &mut t.snode_children,
            t.n_cliques,
        );
    }
}

// -------------------- utilities --------------------

// Given two cliques `c1` and `c2` in the tree `t`, return the parent clique first.

// Not implemented as part of the general SuperNodeTree interface
// since this should only be called when we can guarantee that we
// are acting on a parent-child pair.

fn determine_parent(t: &SuperNodeTree, c1: usize, c2: usize) -> (usize, usize) {
    if t.snode_children[c1].contains(&c2) {
        (c1, c2)
    } else {
        (c2, c1)
    }
}

// not implemented as part of the main SuperNodeTree interface since the
// index is not passed through the post ordering
fn clique_dim(t: &SuperNodeTree, i: usize) -> (usize, usize) {
    (t.snode[i].len(), t.separators[i].len())
}

// Compute the amount of fill-in created by merging two cliques with the
// respective supernode and separator dimensions.

fn fill_in(
    dim_clique_snode: usize,
    dim_clique_sep: usize,
    dim_parent_snode: usize,
    dim_parent_sep: usize,
) -> usize {
    let dim_parent = dim_parent_snode + dim_parent_sep;
    let dim_clique = dim_clique_snode + dim_clique_sep;

    (dim_parent - dim_clique_sep) * (dim_clique - dim_clique_sep)
}
