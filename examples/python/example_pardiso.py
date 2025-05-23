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

# settings this to "mkl" will only work if clarabel has been built 
# with pardiso support (feature "pardiso-mkl"), the mkl library is 
# available and on a path known to the solver (e.g. LD_LIBRARY_PATH)
# or appropriate environment variables are set (e.g. MKLROOT).  Will
# also only work on x86_64 architecture.

settings.direct_solve_method = "mkl"

solver = clarabel.DefaultSolver(P, q, A, b, cones, settings)
solution = solver.solve()
print(
    f"Solver terminated with solution"
    f"{dict(s=solution.s, x=solution.x, z=solution.z)}"
)
