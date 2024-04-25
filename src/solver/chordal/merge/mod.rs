mod clique_graph;
mod disjoint_set_union;
mod nomerge;
mod parent_child;
use crate::solver::chordal::*;
pub(crate) use clique_graph::*;
pub(crate) use disjoint_set_union::*;
pub(crate) use nomerge::*;
pub(crate) use parent_child::*;

// All merge strategies must implement the following trait.

pub(crate) trait MergeStrategy {
    // default implementation for all strategies
    fn merge_cliques(&mut self, t: &mut SuperNodeTree) {
        self.initialise(t);

        while !self.is_done() {
            // find merge candidates
            let Some(cand) = self.traverse(t) else {
                break; //bail if no candidates
            };

            // evaluate wether to merge the candidates
            let do_merge = self.evaluate(t, cand);
            if do_merge {
                self.merge_two_cliques(t, cand);
            }

            // update strategy information after the merge
            self.update_strategy(t, cand, do_merge);

            if t.n_cliques == 1 {
                break;
            }
        }
        self.post_process_merge(t);
    }

    // initialise the tree and strategy
    fn initialise(&mut self, t: &mut SuperNodeTree);

    // merging complete, so stop the merging process
    fn is_done(&self) -> bool;

    // find the next merge candidates
    fn traverse(&mut self, t: &SuperNodeTree) -> Option<(usize, usize)>;

    // evaluate whether to merge a candidate pair or not
    fn evaluate(&mut self, t: &SuperNodeTree, cand: (usize, usize)) -> bool;

    // execute a merge
    fn merge_two_cliques(&self, t: &mut SuperNodeTree, cand: (usize, usize));

    // update the tree/graph and strategy
    fn update_strategy(&mut self, t: &SuperNodeTree, cand: (usize, usize), do_merge: bool);

    // do any post-processing of the tree/graph
    fn post_process_merge(&mut self, t: &mut SuperNodeTree);
}

// PJG: make a settable option
#[derive(Clone, Copy, Debug)]
pub(crate) enum EdgeWeightMethod {
    Cubic = 1,
}

// utilities

// Implements sets[c1] = union(sets[c1],sets[c2]).   This function is
// necessary in Rust since it is awkward to take references to two sets
// from the same array simultaneously.  The equivalent Julia version
// does not require this function.

fn set_union_into_indexed(sets: &mut [VertexSet], c1: usize, c2: usize) {
    if c1 == c2 {
        return;
    }

    // PJG: this function really needs a unit test
    let (target, source);

    if c1 < c2 {
        let (head, tail) = sets.split_at_mut(c1 + 1);
        target = &mut head[c1];
        source = &tail[c2 - c1 - 1];
    } else {
        let (head, tail) = sets.split_at_mut(c2 + 1);
        source = &head[c2];
        target = &mut tail[c1 - c2 - 1];
    }

    for &el in source {
        target.insert(el);
    }
}
