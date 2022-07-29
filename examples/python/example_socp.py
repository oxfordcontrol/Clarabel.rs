import clarabel;
import numpy as np;
from scipy import sparse;

# Define problem data
P = sparse.csc_matrix([[0., 0.], [0., 2.]]);
P = P.tocsc();

q = np.array([0.,0.]);

A = sparse.csc_matrix( \
    [[ 0.,  0.],
     [-2.,  0.],
     [ 0., -1.]]);

b = np.array([1.,-2.,-2.]);

cones = [clarabel.SecondOrderConeT(3)]
settings = clarabel.DefaultSettings();
settings.max_iter = 15;
settings.verbose = True;

solver = clarabel.DefaultSolver(P,q,A,b,cones,settings);
solver.solve()
