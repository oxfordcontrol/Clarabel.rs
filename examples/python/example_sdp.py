import clarabel;
import numpy as np;
from scipy import sparse;

# Define problem data
P = sparse.eye(6).tocsc()
A = sparse.eye(6).tocsc()

c = np.zeros(6);
b = np.array([-3., 1., 4., 1., 2., 5.]);

cones = [clarabel.PSDTriangleConeT(3)]
settings = clarabel.DefaultSettings();

solver = clarabel.DefaultSolver(P,c,A,b,cones,settings);
solution = solver.solve()
print(
    f"Solver terminated with status: {solution.status}, objective {solution.obj_val},\n"
    f"and solution: {dict(s=solution.s, x=solution.x, z=solution.z)}"
)
