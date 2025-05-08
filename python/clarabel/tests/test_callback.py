import clarabel
import pytest
import numpy as np
from scipy import sparse


@pytest.fixture
def callback_qp_data():

    P = sparse.csc_matrix([[4., 1.], [1., 2.]])
    P = sparse.triu(P).tocsc()

    A = sparse.csc_matrix(
        [[-1., -1.],
         [-1.,  0.],
         [0.,  -1.],
         [1.,   1.],
         [1.,   0.],
         [0.,   1.]])

    q = np.array([1., 1.])
    b = np.array([-1., 0., 0., 1., 0.7, 0.7])

    cones = [clarabel.NonnegativeConeT(3), clarabel.NonnegativeConeT(3)]
    settings = clarabel.DefaultSettings()
    return P, q, A, b, cones, settings


@pytest.fixture
def test_callback(callback_qp_data):

    P, q, A, b, cones, settings = callback_qp_data

    def test_termination_fcn(info: clarabel.DefaultInfo) -> bool:
        return info.iterations > 2

    # solves
    solver = clarabel.DefaultSolver(P, q, A, b, cones, settings)
    solver.solve()

    # stops early
    solver.set_termination_callback(test_termination_fcn)
    solver.solve()

    # works again
    solver.unset_termination_callback()
    solver.solve()
