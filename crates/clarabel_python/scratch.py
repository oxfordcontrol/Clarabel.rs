import clarabel_python as clarabel;
import numpy as np
from scipy import sparse

# Define problem data
P = sparse.csc_matrix([[4., 1], [1, 2]])
P = sparse.triu(P).tocsc();
q = np.array([1, 1])
A = sparse.csc_matrix([[1, 1], [1, 0], [0, 1]])
b = np.array([1, 1, 1])

clarabel.solve(P,q,A,b)