import clarabel
import numpy as np
from scipy import sparse


def termination_callback(info: clarabel.DefaultInfo) -> bool:
    """
    Evaluate solver info and return a boolean result.

    Parameters
    ----------
    info : clarabel.DefaultInfo
        The solver info object to evaluate

    Returns
    -------
    bool
        True to signal termination
    """
    if info.iterations < 3:
        print("tick")
        return False  # continue
    else:
        print("BOOM!")
        return True  # stop


P = sparse.csc_matrix([[4., 1.], [1., 2.]])
P = sparse.triu(P).tocsc()

A = sparse.csc_matrix([
    [-1., -1.],
    [-1.,  0.],
    [0.,  -1.],
    [1.,   1.],
    [1.,   0.],
    [0.,   1.]])

q = np.array([1., 1.])
b = np.array([-1., 0., 0., 1., 0.7, 0.7])

cones = [clarabel.NonnegativeConeT(3), clarabel.NonnegativeConeT(3)]
settings = clarabel.DefaultSettings()

# solves
solver = clarabel.DefaultSolver(P, q, A, b, cones, settings)
solver.solve()

# stops early
solver.set_termination_callback(termination_callback)
solver.solve()

# works again
solver.unset_termination_callback()
solver.solve()
