// disjoint set union type for clique graph merge
// See: https://www.cs.princeton.edu/~wayne/kleinberg-tardos/pdf/UnionFind-2x2.pdf

pub(crate) struct DisjointSetUnion {
    parents: Vec<usize>,
    ranks: Vec<usize>,
}

impl DisjointSetUnion {
    pub(crate) fn new(n: usize) -> Self {
        Self {
            parents: (0..n).collect(),
            ranks: vec![0; n],
        }
    }

    pub(crate) fn union(&mut self, x: usize, y: usize) {
        let r = self.root(x);
        let s = self.root(y);

        if r == s {
            return;
        }

        let rankr = self.ranks[r];
        let ranks = self.ranks[s];

        if rankr > ranks {
            self.parents[s] = r;
        } else if rankr < ranks {
            self.parents[r] = s;
        } else {
            self.parents[r] = s;
            self.ranks[s] += 1;
        }
    }

    pub(crate) fn in_same_set(&mut self, x: usize, y: usize) -> bool {
        self.root(x) == self.root(y)
    }

    fn root(&mut self, x: usize) -> usize {
        let mut parent = x;
        while parent != self.parents[x] {
            self.parents[x] = self.parents[self.parents[x]]; //path compression
            parent = self.parents[x];
        }
        parent
    }
}

#[test]
fn test_union() {
    // basic union operations
    let mut dsu = DisjointSetUnion::new(5);
    dsu.union(0, 1);
    dsu.union(2, 3);
    dsu.union(1, 2);
    assert_eq!(dsu.in_same_set(0, 2), true);
    assert_eq!(dsu.in_same_set(1, 3), true);
    assert_eq!(dsu.in_same_set(0, 3), true);
    assert_eq!(dsu.in_same_set(4, 2), false);

    // entry union with itself
    let mut dsu = DisjointSetUnion::new(5);
    dsu.union(0, 0);
    assert_eq!(dsu.in_same_set(0, 0), true);

    // Test union with larger set
    let mut dsu = DisjointSetUnion::new(10);
    dsu.union(0, 1);
    dsu.union(2, 3);
    dsu.union(1, 2);
    dsu.union(0, 4);
    dsu.union(5, 6);
    dsu.union(7, 8);
    dsu.union(4, 6);
    dsu.union(3, 8);
    assert_eq!(dsu.in_same_set(0, 6), true);
    assert_eq!(dsu.in_same_set(2, 7), true);
    assert_eq!(dsu.in_same_set(3, 4), true);
}

#[test]
fn test_in_same_set() {
    // Test elements in the same set
    let mut dsu = DisjointSetUnion::new(5);
    dsu.union(0, 1);
    assert_eq!(dsu.in_same_set(0, 1), true);
    assert_eq!(dsu.in_same_set(1, 0), true);

    // Test elements not in the same set
    assert_eq!(dsu.in_same_set(0, 2), false);
    assert_eq!(dsu.in_same_set(3, 4), false);
}

#[test]
fn test_root() {
    // Test finding root elements
    let mut dsu = DisjointSetUnion::new(5);
    dsu.union(0, 1);
    assert_eq!(dsu.root(1), dsu.root(0));
    assert_eq!(dsu.root(3), 3);

    // Test finding root with larger set
    let mut dsu = DisjointSetUnion::new(10);
    dsu.union(0, 1);
    dsu.union(2, 3);
    dsu.union(1, 2);
    let common = dsu.root(2);
    assert_eq!(dsu.root(0), common);
    assert_eq!(dsu.root(1), common);
    assert_eq!(dsu.root(2), common);
    assert_eq!(dsu.root(3), common);
    assert_eq!(dsu.root(4), 4);
}
