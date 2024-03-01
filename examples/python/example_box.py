import clarabel
import numpy as np
from scipy import sparse

# Define problem data
n = 20000
P = sparse.identity(n).tocsc()

A1 = sparse.identity(n).tocsc()
A2 = -sparse.identity(n).tocsc()
A = sparse.vstack([A1, A2]).tocsc()

q = np.ones(n)
b = np.ones(2*n)

cones = [clarabel.NonnegativeConeT(b.size)]
settings = clarabel.DefaultSettings()

solver = clarabel.DefaultSolver(P, q, A, b, cones, settings)
solution = solver.solve()
