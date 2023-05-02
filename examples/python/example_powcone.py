import clarabel;
import numpy as np;
from scipy import sparse;

# Define problem data
P = sparse.csc_matrix((6,6))

q = np.array([0., 0., -1., 0., 0., -1.])

A = sparse.csc_matrix( \
    [[-1., 0., 0., 0., 0., 0.],
     [0., -1., 0., 0., 0., 0.],
     [0., 0., -1., 0., 0., 0.],
     [0., 0., 0., -1., 0., 0.],
     [0., 0., 0., 0., -1., 0.],
     [0., 0., 0., 0., 0., -1.],
     [1., 2., 0., 3., 0., 0.],
     [0., 0., 0., 0., 1., 0.]]);     

b = np.array([0., 0., 0., 0., 0., 0., 3., 1.])

cones = [clarabel.PowerConeT(0.6), clarabel.PowerConeT(0.1), clarabel.ZeroConeT(2)]
settings = clarabel.DefaultSettings()

solver = clarabel.DefaultSolver(P,q,A,b,cones,settings)
solver.solve()


