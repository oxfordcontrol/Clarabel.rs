#![allow(non_snake_case)]
use super::VertexSet;
use crate::algebra::*;
use std::iter::zip;

// this value used to mark root notes, i.e. ones with no parent
pub(crate) const NO_PARENT: usize = usize::MAX;

// when cliques are merged, their vertices are marked thusly
pub(crate) const INACTIVE_NODE: usize = usize::MAX - 1;

// A structure to represent and analyse the sparsity pattern of an LDL factor matrix L.
#[derive(Debug)]
pub(crate) struct SuperNodeTree {
    // vertices of supernodes stored in one array (also called residuals)
    pub snode: Vec<VertexSet>,
    // post order of supernodal elimination tree
    pub snode_post: Vec<usize>,
    // parent of each supernodes
    pub snode_parent: Vec<usize>,
    // children of each supernode
    pub snode_children: Vec<VertexSet>,
    // post ordering of the vertices in elim tree σ(j) = v
    pub post: Vec<usize>,
    // vertices of clique seperators
    pub separators: Vec<VertexSet>,

    // sizes of submatrices defined by each clique, sorted by post-ordering,
    // e.g. size of clique with order 3 => nblk[3].   Only populated
    // after a post-merging call to `calculate_block_dimensions!`
    pub nblk: Option<Vec<usize>>,

    // number of nonempty supernodes / cliques in tree
    pub(crate) n_cliques: usize,
    //phantom: PhantomData<T>,
}

impl SuperNodeTree {
    pub fn new<T: FloatT>(L: &CscMatrix<T>) -> Self {
        let parent = parent_from_L(L);
        let mut children = children_from_parent(&parent);
        let mut post = vec![0; parent.len()];
        post_order(&mut post, &parent, &mut children, parent.len());

        let degree = higher_degree(L);
        let (snode, snode_parent) = find_supernodes(&parent, &post, &degree);

        let mut snode_children = children_from_parent(&snode_parent);
        let mut snode_post = vec![0; snode_parent.len()];

        post_order(
            &mut snode_post,
            &snode_parent,
            &mut snode_children,
            snode_parent.len(),
        );

        // Her we find separators in all cases, unlike COSMO which defers until
        // after merging for the clique graph merging case.  These are later
        // modified in the clique-graph merge case in the call to add_separators
        let separators = find_separators(L, &snode);

        // nblk will be allocated to the length of the *post-merging*
        // supernode count in calculate_block_dimensions!
        let nblk = None;

        // number of cliques / nonempty supernodes.  decrements as supernodes are merged
        let n_cliques = snode.len();

        Self {
            snode,
            snode_post,
            snode_parent,
            snode_children,
            post,
            separators,
            nblk,
            n_cliques,
        }
    }

    #[allow(dead_code)]
    pub(crate) fn get_post_order(&self, i: usize) -> usize {
        self.snode_post[i]
    }

    pub(crate) fn get_snode(&self, i: usize) -> &VertexSet {
        &self.snode[self.snode_post[i]]
    }

    pub(crate) fn get_separators(&self, i: usize) -> &VertexSet {
        &self.separators[self.snode_post[i]]
    }

    pub(crate) fn get_clique_parent(&self, clique_index: usize) -> usize {
        self.snode_parent[self.snode_post[clique_index]]
    }

    // the block sizes are stored in post order, e.g. if clique 4 (stored in pos 4)
    // has order 2, then nblk[2] represents the cardinality of clique 4
    // should onlt be called after block sizes are populated
    pub(crate) fn get_nblk(&self, i: usize) -> usize {
        self.nblk.as_ref().unwrap()[i]
    }

    pub(crate) fn get_overlap(&self, i: usize) -> usize {
        self.separators[self.snode_post[i]].len()
    }

    pub(crate) fn get_clique(&self, i: usize) -> VertexSet {
        let c = self.snode_post[i];
        let set1 = &self.snode[c];
        let set2 = &self.separators[c];

        let mut out = VertexSet::with_capacity(set1.len() + set2.len());
        set1.union(set2).for_each(|&v| {
            out.insert(v);
        });
        out
    }

    pub(crate) fn get_decomposed_dim_and_overlaps(&self) -> (usize, usize) {
        let mut dim = 0;
        let mut overlaps = 0;
        for i in 0..self.n_cliques {
            dim += triangular_number(self.get_nblk(i));
            overlaps += triangular_number(self.get_overlap(i));
        }
        (dim, overlaps)
    }

    // Takes a SuperNodeTree and reorders the vertices in each supernode (and separator) to have consecutive order.

    // The reordering is needed to achieve equal column structure for the psd completion of the dual variable `Y`.
    // This also modifies `ordering` which maps the vertices in the `sntree` back to the actual location in the
    // not reordered data, i.e. the primal constraint variable `S` and dual variables `Y`.

    pub(crate) fn reorder_snode_consecutively(&mut self, ordering: &mut [usize]) {
        // determine permutation vector p and permute the vertices in each snd
        let mut p = vec![0; self.post.len()];

        let mut k = 0;

        for &i in self.snode_post.iter() {
            let snode = &mut self.snode[i];
            let n = snode.len();
            for (j, &v) in snode.iter().enumerate() {
                p[j + k] = v;
            }
            p[k..(k + n)].sort();

            // assign k..(k+n) to the OrderedSet snode,
            // dropping the previous values
            snode.clear();
            snode.extend(k..(k + n));

            k += n;
        }

        // permute the separators as well
        let p_inv = invperm(&p);

        for sp in self.separators.iter_mut() {
            // use here the permutation vector p as scratch before flushing
            // the separator set and repopulating.  Assumes that the permutation
            // will be at least as long as the largest separator set

            assert!(p.len() >= sp.len());
            let tmp = &mut p[0..sp.len()];
            for (i, &x) in sp.iter().enumerate() {
                tmp[i] = p_inv[x];
            }
            sp.clear();
            sp.extend(tmp.iter());
        }

        // because I used 'p' as scratch space, I will
        // ipermute using pinv rather than permute using p
        let tmp = ordering.to_vec(); //allocate a copy
        ipermute(ordering, &tmp, &p_inv);
    }

    pub(crate) fn calculate_block_dimensions(&mut self) {
        let n = self.n_cliques;
        let mut nblk = vec![0; n];

        for i in 0..n {
            let c = self.snode_post[i];
            nblk[i] = self.separators[c].len() + self.snode[c].len();
        }
        self.nblk = Some(nblk);
    }
}

// -------------------------
// utility functions for SuperNodeTree

fn parent_from_L<T>(L: &CscMatrix<T>) -> Vec<usize>
where
    T: FloatT,
{
    let mut parent = vec![NO_PARENT; L.n];
    // loop over vertices of graph
    for (i, par) in parent.iter_mut().enumerate() {
        *par = find_parent_direct(L, i);
    }
    parent
}

fn find_parent_direct<T>(L: &CscMatrix<T>, v: usize) -> usize
where
    T: FloatT,
{
    if v == L.nrows() - 1 {
        return NO_PARENT;
    }

    L.rowval[L.colptr[v]]
}

fn find_separators<T>(L: &CscMatrix<T>, snode: &[VertexSet]) -> Vec<VertexSet>
where
    T: FloatT,
{
    let mut separators = new_vertex_sets(snode.len());

    for (sn, sep) in zip(snode, separators.iter_mut()) {
        let vrep = *sn.iter().min().unwrap();
        let adjplus = find_higher_order_neighbors(L, vrep);

        for neighbor in adjplus {
            if !sn.contains(neighbor) {
                sep.insert(*neighbor);
            }
        }
    }
    separators
}

fn find_higher_order_neighbors<T>(L: &CscMatrix<T>, v: usize) -> &[usize] {
    &L.rowval[L.colptr[v]..(L.colptr[v + 1])]
}

fn higher_degree<T>(L: &CscMatrix<T>) -> Vec<usize>
where
    T: FloatT,
{
    let mut degree = vec![0usize; L.ncols()];
    for v in 0..(L.n - 1) {
        degree[v] = L.colptr[v + 1] - L.colptr[v];
    }
    degree
}

fn children_from_parent(parent: &[usize]) -> Vec<VertexSet> {
    let mut children = new_vertex_sets(parent.len());
    for (i, &pi) in parent.iter().enumerate() {
        if pi != NO_PARENT {
            children[pi].insert(i);
        }
    }
    children
}

pub(crate) fn post_order(
    post: &mut Vec<usize>,
    parent: &[usize],
    children: &mut [VertexSet],
    nc: usize,
) {
    let mut order = vec![nc + 1; parent.len()];

    let root = parent.iter().position(|&x| x == NO_PARENT).unwrap(); //should always be a root

    let mut stack = Vec::with_capacity(parent.len());
    stack.push(root);

    post.resize(parent.len(), 0);
    post.iter_mut().enumerate().for_each(|(i, p)| *p = i);

    let mut i = nc;

    while let Some(v) = stack.pop() {
        //not empty in loop
        order[v] = i;
        i -= 1;

        // maybe faster to append to the stack vector and then
        // sort a view of what was added, but this way gets
        // the children sorted and keeps everything consistent
        // with the COSMO implementation for reference
        children[v].sort();
        stack.extend(children[v].iter());
    }

    post.sort_by(|&x, &y| order[x].cmp(&order[y]));

    if nc != parent.len() {
        post.truncate(nc);
    }
}

fn find_supernodes(
    parent: &[usize],
    post: &[usize],
    degree: &[usize],
) -> (Vec<VertexSet>, Vec<usize>) {
    let mut snode = new_vertex_sets(parent.len());

    let (snode_parent, snode_index) = pothen_sun(parent, post, degree);

    for (i, &f) in snode_index.iter().enumerate() {
        if f < 0 {
            snode[i].insert(i);
        } else {
            snode[f as usize].insert(i);
        }
    }
    snode.retain(|x| !x.is_empty());
    (snode, snode_parent)
}

fn pothen_sun(parent: &[usize], post: &[usize], degree: &[usize]) -> (Vec<usize>, Vec<isize>) {
    let n = parent.len();

    // if snode_index[v] < 0 then v is a rep vertex, otherwise v ∈ supernode[snode_index[v]]
    // NB: snode_index is never actually used as an index into anything, so
    // ok to keep it as Rust isize.

    let mut snode_index = vec![-1isize; n];
    let mut snode_parent = vec![NO_PARENT; n];

    // This also works as array of Int[], which might be faster
    // note this arrays is local to the function, not the one
    // contained in the SuperNodeTree
    let mut children = new_vertex_sets(parent.len());

    // find the root
    let root_index = parent.iter().position(|&x| x == NO_PARENT).unwrap();

    // go through parents of vertices in post_order
    for &v in post {
        if parent[v] == NO_PARENT {
            children[root_index].insert(v);
        } else {
            children[parent[v]].insert(v);
        }

        // parent is not the root.
        if parent[v] != NO_PARENT {
            if degree[v] - 1 == degree[parent[v]] && snode_index[parent[v]] == -1 {
                // Case A: v is a representative vertex
                if snode_index[v] < 0 {
                    snode_index[parent[v]] = v as isize;
                    snode_index[v] -= 1;
                }
                // Case B: v is not representative vertex, add to sn_ind[v] instead
                else {
                    snode_index[parent[v]] = snode_index[v];
                    let tmp = snode_index[v] as usize;
                    snode_index[tmp] -= 1;
                }
            } else if snode_index[v] < 0 {
                snode_parent[v] = v;
            } else {
                snode_parent[snode_index[v] as usize] = snode_index[v] as usize;
            }
        }

        // k: rep vertex of the snd that v belongs to
        let k: isize = {
            if snode_index[v] < 0 {
                v as isize
            } else {
                snode_index[v]
            }
        };
        // loop over v's children
        let v_children = &children[v];

        if !v_children.is_empty() {
            for &w in v_children {
                let l = {
                    if snode_index[w] < 0 {
                        w
                    } else {
                        snode_index[w] as usize
                    }
                };
                if l != (k as usize) {
                    snode_parent[l] = k as usize
                }
            }
        }
    } // loop over vertices

    // representative vertices
    let repr_vertex = snode_index.iter().position_all(|&x| *x < 0);

    // vertices that are the parent of representative vertices
    let repr_parent: Vec<usize> = repr_vertex.iter().map(|&i| snode_parent[i]).collect();

    // resize and reset snode_parent to take into account that all
    // non-representative arrays are removed from the parent structure
    snode_parent.clear();
    snode_parent.resize(repr_vertex.len(), NO_PARENT);

    for (i, &rp) in repr_parent.iter().enumerate() {
        let rpidx = repr_vertex.iter().position(|&x| x == rp);

        match rpidx {
            Some(rpidx) => snode_parent[i] = rpidx,
            None => snode_parent[i] = NO_PARENT,
        }
    }

    (snode_parent, snode_index)
}

fn new_vertex_sets(n: usize) -> Vec<VertexSet> {
    (0..n).map(|_| VertexSet::new()).collect()
}
