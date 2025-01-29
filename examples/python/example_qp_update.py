import clarabel
import numpy as np
from scipy import sparse

# Define problem data
P = sparse.csc_matrix([[6., 0.], [0., 4.]])
P = sparse.triu(P).tocsc()

q = np.array([-1., -4.])

A = sparse.csc_matrix(
    [[1., -2.],        # <-- LHS of equality constraint (lower bound)
     [1.,  0.],        # <-- LHS of inequality constraint (upper bound)
     [0.,  1.],        # <-- LHS of inequality constraint (upper bound)
     [-1.,  0.],       # <-- LHS of inequality constraint (lower bound)
     [0., -1.]])       # <-- LHS of inequality constraint (lower bound)

b = np.array([0., 1., 1., 1., 1.])

cones = [clarabel.ZeroConeT(1), clarabel.NonnegativeConeT(4)]
settings = clarabel.DefaultSettings()
settings.presolve_enable = False

solver = clarabel.DefaultSolver(P, q, A, b, cones, settings)

# complete vector data overwrite
qnew = np.array([0., 0.])

# partial vector data update
bv = np.array([0., 1.])
bi = np.array([1, 2])
bnew = (bi, bv)

# complete matrix data overwrite
Pnew = sparse.csc_matrix([[3., 0.], [0., 4.]]).tocsc()

# complete matrix data update (vector of nonzero values)
# NB: tuple of partial updates also works
Anew = A.data.copy()
Anew[1] = 2.

solver.update(q=qnew, P=Pnew, b=bnew, A=Anew)
