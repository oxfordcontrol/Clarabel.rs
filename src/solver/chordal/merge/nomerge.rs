// All merge strategies must implement the following trait.

use crate::solver::chordal::*;

pub(crate) struct NoMergeStrategy;

impl NoMergeStrategy {
    pub(crate) fn new() -> Self {
        Self {}
    }
}

impl MergeStrategy for NoMergeStrategy {
    fn initialise(&mut self, _t: &mut SuperNodeTree) {
        //no-op
    }

    fn is_done(&self) -> bool {
        true
    }

    fn traverse(&mut self, _t: &SuperNodeTree) -> Option<(usize, usize)> {
        unreachable!()
    }

    fn evaluate(&mut self, _t: &SuperNodeTree, _cand: (usize, usize)) -> bool {
        unreachable!()
    }

    fn merge_two_cliques(&self, _t: &mut SuperNodeTree, _cand: (usize, usize)) {
        unreachable!()
    }

    fn update_strategy(&mut self, _t: &SuperNodeTree, _cand: (usize, usize), _do_merge: bool) {
        unreachable!()
    }

    fn post_process_merge(&mut self, _t: &mut SuperNodeTree) {
        //no-op
    }
}
