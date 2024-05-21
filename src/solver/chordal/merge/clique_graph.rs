#![allow(non_snake_case)]

use super::*;
use crate::algebra::*;
use crate::solver::chordal::*;
use std::cmp::{max, min, Reverse};
use std::collections::HashMap;
use std::iter::zip;

// The (default) merge strategy based on the *reduced* clique graph ``\\mathcal{G}(\\mathcal{B}, \\xi)``,
// for a set of cliques ``\\mathcal{B} = \\{ \\mathcal{C}_1, \\dots, \\mathcal{C}_p\\}``, where the edge
// set ``\\xi`` is obtained by taking the edges of the union of clique trees.

// Moreover, given an edge weighting function ``e(\\mathcal{C}_i,\\mathcal{C}_j) = w_{ij}``, we compute a
// weight for each edge that quantifies the computational savings of merging the two cliques.
//
// After the initial weights are computed, we merge cliques in a loop:
//
// **while** clique graph contains positive weights:
// - select two permissible cliques with the highest weight ``w_{ij}``
// - merge cliques ``\\rightarrow`` update clique graph
// - recompute weights for updated clique graph
//
// See also: *Garstka, Cannon, Goulart - A clique graph based merging strategy for decomposable SDPs (2019)*
//
// NB: edges is currently an integer valued matrix since the weights are taken as
// powers of the cardinality of the intersection of the cliques. This needs to change
// to floats if empirical edge weight functions are to be supported.

pub(crate) struct CliqueGraphMergeStrategy {
    stop: bool,              // a flag to indicate that merging should be stopped
    edges: CscMatrix<isize>, // the edges and weights of the reduced clique graph
    p: Vec<usize>,           // as a workspace variable to store the sorting of weights
    adjacency_table: HashMap<usize, VertexSet>, // a double structure of edges, to allow fast lookup of neighbors
    edge_weight: EdgeWeightMethod, // used to dispatch onto the correct scoring function
}

impl CliqueGraphMergeStrategy {
    pub(crate) fn new() -> Self {
        Self {
            stop: false,
            edges: CscMatrix::zeros((0, 0)),
            p: Vec::new(),
            adjacency_table: HashMap::new(),
            edge_weight: EdgeWeightMethod::Cubic, //PJG: make settable
        }
    }
}

impl MergeStrategy for CliqueGraphMergeStrategy {
    fn initialise(&mut self, t: &mut SuperNodeTree) {
        // this merge strategy is clique-graph based, we give up the tree structure and add
        // the seperators to the supernodes.  The supernodes then represent the full clique.
        // after clique merging a new clique tree will be computed in post_process_merge!
        // for this type

        for (snode, separator) in zip(t.snode.iter_mut(), t.separators.iter()) {
            for &s in separator {
                snode.insert(s);
            }
        }

        for i in 0..t.snode_parent.len() {
            t.snode_parent[i] = INACTIVE_NODE;
            t.snode_children[i] = VertexSet::new();
        }

        // compute the edges and intersections of cliques in the reduced clique graph
        let (rows, cols) = compute_reduced_clique_graph(&mut t.separators, &t.snode);

        let weights = compute_weights(&rows, &cols, &t.snode, self.edge_weight);

        self.edges = CscMatrix::new_from_triplets(t.n_cliques, t.n_cliques, rows, cols, weights);
        self.p = vec![0; self.edges.nzval.len()];
        self.adjacency_table = compute_adjacency_table(&self.edges, t.n_cliques);
    }

    fn is_done(&self) -> bool {
        self.stop
    }

    fn traverse(&mut self, t: &SuperNodeTree) -> Option<(usize, usize)> {
        let p = &mut self.p;

        // find edge with highest weight, if permissible return cliques
        let edge = max_elem(&self.edges);

        if ispermissible(edge, &self.adjacency_table, &t.snode) {
            return Some(edge);
        }

        // sort the weights in edges.nzval to find the permutation p
        let slicep = &mut p[0..self.edges.nzval.len()];
        sortperm_rev(slicep, &self.edges.nzval);

        // try edges with decreasing weight and check if the edge is permissible
        // PJG: potentially returns nothing in Julia version?
        for k in 1..self.edges.nzval.len() {
            let edge = edge_from_index(&self.edges, p[k]);

            if ispermissible(edge, &self.adjacency_table, &t.snode) {
                return Some(edge);
            }
        }

        None
    }

    fn evaluate(&mut self, _t: &SuperNodeTree, cand: (usize, usize)) -> bool {
        let (c1, c2) = cand;

        let do_merge = self.edges.get_entry((c1, c2)).unwrap() >= 0;

        if !do_merge {
            self.stop = true;
        }
        do_merge
    }

    fn merge_two_cliques(&self, t: &mut SuperNodeTree, cand: (usize, usize)) {
        let (c1, c2) = cand;

        // merge clique c2 into c1
        set_union_into_indexed(&mut t.snode, c1, c2);
        t.snode[c2].clear();

        // decrement number of mergeable / nonempty cliques in graph
        t.n_cliques -= 1
    }

    fn update_strategy(&mut self, t: &SuperNodeTree, cand: (usize, usize), do_merge: bool) {
        if !do_merge {
            return;
        }

        // After a merge operation update the information of the strategy

        let (c_1_ind, c_removed) = cand;

        let edges = &mut self.edges;
        let n = edges.ncols();
        let adjacency_table = &mut self.adjacency_table;

        let c_1 = &t.snode[c_1_ind];
        let neighbors = &adjacency_table[&c_1_ind];

        // neighbors exclusive to the removed clique (and not c1)
        // order preserving removal for consistency (?) with Julia
        let mut new_neighbors = adjacency_table[&c_removed].clone();
        for e in neighbors.iter() {
            new_neighbors.shift_remove(e);
        }
        new_neighbors.shift_remove(&c_1_ind);

        // recalculate edge values of all of c_1's neighbors
        for &n_ind in neighbors {
            if n_ind != c_removed {
                let neighbor = &t.snode[n_ind];
                let row = max(c_1_ind, n_ind);
                let col = min(c_1_ind, n_ind);
                let val = edge_metric(c_1, neighbor, self.edge_weight);
                edges.set_entry((row, col), val);
            }
        }

        // point edges exclusive to removed clique to surviving clique 1
        for &n_ind in new_neighbors.iter() {
            let neighbor = &t.snode[n_ind];
            let row = max(c_1_ind, n_ind);
            let col = min(c_1_ind, n_ind);
            let val = edge_metric(c_1, neighbor, self.edge_weight);
            edges.set_entry((row, col), val);
        }

        // overwrite the weight to any removed edges that still contain a link to c_removed
        for row in (c_removed + 1)..n {
            edges.set_entry((row, c_removed), 0);
        }
        for col in 0..c_removed {
            edges.set_entry((c_removed, col), 0);
        }
        edges.dropzeros();

        // update adjacency table in a similar manner
        for new_neighbor in new_neighbors.iter() {
            adjacency_table
                .get_mut(&c_1_ind)
                .unwrap()
                .insert(*new_neighbor);
            adjacency_table
                .get_mut(new_neighbor)
                .unwrap()
                .insert(c_1_ind);
        }

        adjacency_table.remove(&c_removed);

        for set in adjacency_table.values_mut() {
            set.shift_remove(&c_removed);
        }
    }

    fn post_process_merge(&mut self, t: &mut SuperNodeTree) {
        // since for now we have a graph, not a tree, a post ordering or a parent structure
        // does not make sense. Therefore just number the non-empty supernodes in t.snd

        t.snode_post = t.snode.iter().position_all(|&x| !x.is_empty());
        t.snode_parent = vec![INACTIVE_NODE; t.snode.len()];

        // recompute a clique tree from the clique graph
        if t.n_cliques > 1 {
            self.clique_tree_from_graph(t);
        }

        // PJG: This seems unnecessary because the next operation on this
        // object is the call to reorder_snode_consecutively, which overwrites
        // the snode anyway.  Treatment of separators possibly ends up different.
        // Seems to work without, but keep for now for consistency with COSMO.

        t.snode.iter_mut().for_each(|s| s.sort());
        t.separators.iter_mut().for_each(|s| s.sort());
    }
}

impl CliqueGraphMergeStrategy {
    fn clique_tree_from_graph(&mut self, t: &mut SuperNodeTree) {
        // a clique tree is a maximum weight spanning tree of the clique graph, where the edge weight is the
        // cardinality of the intersection between two cliques compute intersection value for each edge
        // in the clique graph

        clique_intersections(&mut self.edges, &t.snode);

        // find a maximum weight spanning tree of the clique graph using Kruskal's algorithm
        kruskal(&mut self.edges, t.n_cliques);

        // determine the root clique of the clique tree (it can be any clique, but we use the
        // clique that contains the vertex with the highest order)
        determine_parent_cliques(
            &mut t.snode_parent,
            &mut t.snode_children,
            &t.snode,
            &t.post,
            &self.edges,
        );

        // recompute a postorder for the supernodes (NB: snode_post will shrink
        // to the possibly reduced length n_cliques after the merge)
        post_order(
            &mut t.snode_post,
            &t.snode_parent,
            &mut t.snode_children,
            t.n_cliques,
        );

        // Clear the (graph) separators.  They will be rebuilt in the split_cliques
        t.separators.iter_mut().for_each(|set| set.clear());

        // split clique sets back into separators and supernodes
        split_cliques(
            &mut t.snode,
            &mut t.separators,
            &t.snode_parent,
            &t.snode_post,
            t.n_cliques,
        );
    }
}

// ------------------- internal utilities -------------------

// Compute the reduced clique graph (union of all clique trees) given an initial clique tree defined by its
// supernodes and separator sets.

// We are using the algorithm described in **Michel Habib and Juraj Stacho - Polynomial-time algorithm for the
// leafage of chordal graphs (2009)**, which
// computes the reduced clique graph in the following way:
// 1. Sort all minimal separators by size
// 2. Initialise graph CG(R) with cliques as nodes and no edges
// 3. for largest unprocessed separator S and
//     |  add an edge between any two cliques C1 and C2 if they both contain S and are in different connected
//        components of CG(R) and store in `edges`.
//     |  Compute an edge weight used for merge decision and store in `val`.
//     |  Store the index of the separator which is the intersection C1 ∩ C2 in `iter`
//    end

fn compute_reduced_clique_graph(
    separators: &mut [VertexSet],
    snode: &[VertexSet],
) -> (Vec<usize>, Vec<usize>) {
    // loop over separators by decreasing cardinality
    separators.sort_by_key(|b| Reverse(b.len()));

    let mut rows = Vec::new();
    let mut cols = Vec::new();

    for separator in separators {
        // find cliques that contain the separator
        let clique_indices = snode.iter().position_all(|&x| separator.is_subset(x));

        // Compute the separator graph (see Habib, Stacho - Reduced clique graphs of chordal graphs)
        // to analyse connectivity.  We represent the separator graph H by a hashtable
        let H = separator_graph(&clique_indices, separator, snode);

        // find the connected components of H
        let components = find_components(&H, &clique_indices);

        // for each pair of cliques that contain the separator, add an edge to the reduced
        // clique tree if they are in unconnected components

        let ncliques = clique_indices.len();

        for i in 0..ncliques {
            for j in (i + 1)..ncliques {
                let pair = (clique_indices[i], clique_indices[j]);
                if is_unconnected(pair, &components) {
                    rows.push(max(pair.0, pair.1));
                    cols.push(min(pair.0, pair.1));
                }
            }
        }
    }

    (rows, cols)
}

// Find the separator graph H given a separator and the relevant index-subset of cliques.

fn separator_graph(
    clique_ind: &[usize],
    separator: &VertexSet,
    snd: &[VertexSet],
) -> HashMap<usize, Vec<usize>> {
    // make the separator graph using a hash table
    // key: clique_ind --> edges to other clique indices
    let mut H = HashMap::<usize, Vec<usize>>::new();

    let nindex = clique_ind.len();

    for i in 0..nindex {
        for j in (i + 1)..nindex {
            let ca = &clique_ind[i];
            let cb = &clique_ind[j];
            // if intersect_dim(snd[ca], snd[cb]) > length(separator)
            if !inter_equal(&snd[*ca], &snd[*cb], separator) {
                if H.contains_key(ca) {
                    H.get_mut(ca).unwrap().push(*cb);
                } else {
                    H.insert(*ca, vec![*cb]);
                }
                if H.contains_key(cb) {
                    H.get_mut(cb).unwrap().push(*ca);
                } else {
                    H.insert(*cb, vec![*ca]);
                }
            }
        }
    }
    // add unconnected cliques
    for v in clique_ind.iter() {
        if !H.contains_key(v) {
            H.insert(*v, Vec::new());
        }
    }
    H
}

// Find connected components in undirected separator graph represented by `H`.
fn find_components(H: &HashMap<usize, Vec<usize>>, clique_ind: &[usize]) -> Vec<VertexSet> {
    let mut visited = HashMap::<usize, bool>::with_capacity(clique_ind.len());
    for v in clique_ind {
        visited.insert(*v, false);
    }

    let mut components = Vec::<VertexSet>::new();
    for v in clique_ind {
        if !(*visited.get(v).unwrap()) {
            let mut component = VertexSet::new();
            DFS_hashtable(&mut component, *v, &mut visited, H);
            components.push(component);
        }
    }
    components
}

// Check whether the `pair` of cliques are in different `components`.
fn is_unconnected(pair: (usize, usize), components: &[VertexSet]) -> bool {
    let component_ind = components.iter().position(|x| x.contains(&pair.0)).unwrap();
    !components[component_ind].contains(&pair.1)
}

// Depth first search on a HashMap `H`.
fn DFS_hashtable<'a>(
    component: &'a mut VertexSet,
    v: usize,
    visited: &'a mut HashMap<usize, bool>,
    H: &'a HashMap<usize, Vec<usize>>,
) {
    visited.insert(v, true);
    component.insert(v);
    for n in H.get(&v).unwrap().iter() {
        if !(*visited.get(n).unwrap()) {
            DFS_hashtable(component, *n, visited, H);
        }
    }
}

// Check if s ∩ s2 == s3.
fn inter_equal(s1: &VertexSet, s2: &VertexSet, s3: &VertexSet) -> bool {
    let mut dim = 0;

    let len_s1 = s1.len();
    let len_s2 = s2.len();
    let len_s3 = s3.len();

    // maximum possible intersection size
    let mut max_intersect = len_s1 + len_s2;

    // abort if there's no way the intersection can be the same
    if max_intersect < len_s3 {
        return false;
    }

    let (sa, sb) = {
        if len_s1 < len_s2 {
            (s1, s2)
        } else {
            (s2, s1)
        }
    };

    for e in sa.iter() {
        if sb.contains(e) {
            dim += 1;
            if dim > len_s3 {
                return false;
            }
            if !s3.contains(e) {
                return false;
            }
        }
        max_intersect -= 1;
        if max_intersect < len_s3 {
            return false;
        }
    }
    dim == len_s3
}

// Given a list of edges, return an adjacency hash-table `table` with nodes from 1 to `num_vertices`.

fn compute_adjacency_table(
    edges: &CscMatrix<isize>,
    num_vertices: usize,
) -> HashMap<usize, VertexSet> {
    let mut table = HashMap::<usize, VertexSet>::with_capacity(num_vertices);

    for i in 0..num_vertices {
        table.insert(i, VertexSet::new());
    }

    let r = &edges.rowval;
    let c = &edges.colptr;

    for col in 0..num_vertices {
        for &row in &r[c[col]..c[col + 1]] {
            table.get_mut(&row).unwrap().insert(col);
            table.get_mut(&col).unwrap().insert(row);
        }
    }
    table
}

// Check whether `edge` is permissible for a merge. An edge is permissible if for every common neighbor N,
// C_1 ∩ N == C_2 ∩ N or if no common neighbors exist.

fn ispermissible(
    edge: (usize, usize),
    adjacency_table: &HashMap<usize, VertexSet>,
    snode: &[VertexSet],
) -> bool {
    let (c_1, c_2) = edge;

    let common_neighbors = adjacency_table[&c_1].intersection(&adjacency_table[&c_2]);

    // N.B. This is allocating and could be made more efficient
    for &neighbor in common_neighbors {
        let int1 = snode[c_1].intersection(&snode[neighbor]);
        let int2 = snode[c_2].intersection(&snode[neighbor]);
        if !int1.eq(int2) {
            return false;
        }
    }
    true
}

// Find the matrix indices (i, j) of the first maximum element among the elements stored in A.nzval

fn max_elem(A: &CscMatrix<isize>) -> (usize, usize) {
    let n = A.ncols();

    let ind = findmax(&A.nzval).unwrap();
    let row = A.rowval[ind];

    let mut col = 0;
    for c in 0..n {
        let col_indices = A.colptr[c]..A.colptr[c + 1];
        if col_indices.contains(&ind) {
            col = c;
            break;
        }
    }
    (row, col)
}

fn edge_from_index(A: &CscMatrix<isize>, ind: usize) -> (usize, usize) {
    A.index_to_coord(ind)
}

fn clique_intersections(E: &mut CscMatrix<isize>, snd: &[VertexSet]) {
    // iterate over the nonzeros of the connectivity matrix E which represents the
    // clique graph and replace the value by |C_i ∩ C_j|
    let rows = &E.rowval;

    for col in 0..E.ncols() {
        for j in E.colptr[col]..E.colptr[col + 1] {
            let row = rows[j];
            E.nzval[j] = intersect_dim(&snd[row], &snd[col]) as isize;
        }
    }
}

// Return the number of elements in s ∩ s2.
fn intersect_dim(s1: &VertexSet, s2: &VertexSet) -> usize {
    let (sa, sb) = {
        if s1.len() < s2.len() {
            (s1, s2)
        } else {
            (s2, s1)
        }
    };

    let mut dim = 0;
    for e in sa {
        if sb.contains(e) {
            dim += 1;
        }
    }
    dim
}

// Find the size of the set `A ∪ B` under the assumption that `A` and `B` only have unique elements.
fn union_dim(s1: &VertexSet, s2: &VertexSet) -> usize {
    s1.len() + s2.len() - intersect_dim(s1, s2)
}

// Kruskal's algorithm to find a maximum weight spanning tree from the clique intersection graph.
//
//  `E[i,j]` holds the cardinalities of the intersection between two cliques (i, j). Changes the entries in the
//   connectivity matrix `E` to a negative value if an edge between two cliques is included in the max spanning tree.
//
//  This is a modified version of https://github.com/JuliaGraphs/LightGraphs.jl/blob/master/src/spanningtrees/kruskal.jl

fn kruskal(E: &mut CscMatrix<isize>, num_cliques: usize) {
    let num_initial_cliques = E.ncols();
    let mut connected_c = DisjointSetUnion::new(num_initial_cliques);

    let (I0, J0, V0) = E.findnz();

    // sort the weights and edges from maximum to minimum value
    let mut p = vec![0; V0.len()];
    sortperm_rev(&mut p, &V0);

    let mut I = vec![0; p.len()];
    let mut J = vec![0; p.len()];
    permute(&mut I, &I0, &p);
    permute(&mut J, &J0, &p);

    let mut num_edges_found = 0;

    // iterate through edges (I -- J) with decreasing weight
    for (k, (row, col)) in zip(I, J).enumerate() {
        if !connected_c.in_same_set(row, col) {
            connected_c.union(row, col);
            // indicate an edge in the MST with a negative value in E (all other values are >= 0)
            E.nzval[p[k]] = -1;
            num_edges_found += 1;
            //break when all cliques are connected in one tree
            if num_edges_found >= (num_cliques - 1) {
                break;
            }
        }
    }
}

// Given the maximum weight spanning tree represented by `E`, determine a parent
// structure `snd_par` for the clique tree.

fn determine_parent_cliques(
    snode_parent: &mut [usize],
    snode_children: &mut [VertexSet],
    cliques: &[VertexSet],
    post: &[usize],
    E: &CscMatrix<isize>,
) {
    // vertex with highest order
    let v = post.last().unwrap();
    let mut c = 0;

    // Find clique that contains that vertex
    for (k, clique) in cliques.iter().enumerate() {
        if clique.contains(v) {
            // set that clique to the root
            snode_parent[k] = NO_PARENT;
            c = k;
            break;
        }
    }

    // assign children to cliques along the MST defined by E
    assign_children(snode_parent, snode_children, c, E);
}

fn assign_children(
    snode_parent: &mut [usize],
    snode_children: &mut [VertexSet],
    c: usize,
    edges: &CscMatrix<isize>,
) {
    let mut stack = vec![c];

    while let Some(c) = stack.pop() {
        let neighbors = find_neighbors(edges, c);

        for n in neighbors {
            // conditions that there is a edge in the MST and that n is not the parent of c
            if edges.get_entry((max(c, n), min(c, n))).unwrap_or(0) == -1 && snode_parent[c] != n {
                snode_parent[n] = c;
                snode_children[c].insert(n);
                stack.push(n);
            }
        }
    }
}

// Find all the cliques connected to `c` which are given by the nonzeros in `(c, 1:c-1)` and `(c+1:n, c)`.

fn find_neighbors(edges: &CscMatrix<isize>, c: usize) -> Vec<usize> {
    let mut neighbors = Vec::<usize>::new();
    let (_, n) = edges.size();
    // find all nonzero columns in row c up to column c
    if c > 0 {
        for col in 0..c {
            let val = edges.get_entry((c, col)).unwrap_or(0);
            if val != 0 {
                neighbors.push(col);
            }
        }
    }
    // find all nonzero entries in column c below c
    if c < (n - 1) {
        let rows = &edges.rowval[edges.colptr[c]..edges.colptr[c + 1]];
        if edges.colptr[c] < edges.colptr[c + 1] {
            neighbors.extend(rows);
        }
    }

    neighbors
}

// Traverse the clique tree in descending topological order and split the clique sets into supernodes and separators.

fn split_cliques(
    snode: &mut [VertexSet],
    separators: &mut [VertexSet],
    snode_parent: &[usize],
    snode_post: &[usize],
    num_cliques: usize,
) {
    // travese in topological decending order through the clique tree and split the clique
    // into supernodes and separators
    for j in 0..(num_cliques - 1) {
        let c_ind = snode_post[j];
        let p_ind = snode_parent[c_ind];

        // find intersection of clique with parent
        separators[c_ind] = VertexSet::new();
        separators[c_ind].extend(snode[c_ind].intersection(&snode[p_ind]));

        let mut tmp = VertexSet::new();
        tmp.extend(
            snode[c_ind]
                .iter()
                .filter(|&s| !separators[c_ind].contains(s)),
        );
        snode[c_ind] = tmp;
    }
}

// -------------------
// functions relating to edge weights
// -------------------

// Compute the edge weight between all cliques specified by the edges (rows, cols).
// weights on the edges currently defined as integer values, but could be changed
// to floats to allow emperical edge weight functions.

fn compute_weights(
    rows: &[usize],
    cols: &[usize],
    snode: &[VertexSet],
    edge_weight: EdgeWeightMethod,
) -> Vec<isize> {
    let mut weights = vec![0; rows.len()];

    for k in 0..rows.len() {
        let c_1 = &snode[rows[k]];
        let c_2 = &snode[cols[k]];
        weights[k] = edge_metric(c_1, c_2, edge_weight);
    }
    weights
}

// Given two cliques `c_a` and `c_b` return a value for their edge weight.

fn edge_metric(c_a: &VertexSet, c_b: &VertexSet, edge_weight: EdgeWeightMethod) -> isize {
    let n_1 = c_a.len() as isize;
    let n_2 = c_b.len() as isize;

    // merged block size
    let n_m = union_dim(c_a, c_b) as isize;

    match edge_weight {
        EdgeWeightMethod::Cubic => n_1.pow(3) + n_2.pow(3) - n_m.pow(3),
    }
}
