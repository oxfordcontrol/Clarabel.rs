import clarabel;
import numpy as np;
from scipy import sparse;

# Exponential cone example
# max  x
# s.t. y * exp(x / y) <= z
#      y == 1, z == exp(5)

# Define problem data
P = sparse.csc_matrix((3,3))

q = np.array([-1.,0.,0.])

A = sparse.csc_matrix( \
    [[ -1., 0., 0.],        
     [  0.,-1., 0],        
     [  0., 0.,-1.],        
     [  0., 1., 0.],       
     [  0., 0., 1.]]);     

b = np.array([0.,0.,0.,1.,np.exp(5.)])

cones = [clarabel.ExponentialConeT(), clarabel.ZeroConeT(2)]
settings = clarabel.DefaultSettings()

solver = clarabel.DefaultSolver(P,q,A,b,cones,settings)
solver.solve()
