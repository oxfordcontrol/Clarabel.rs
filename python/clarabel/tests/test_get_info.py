import clarabel
import pytest
import numpy as np
from scipy import sparse


@pytest.fixture
def get_info_qp_data():

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


def test_get_info(get_info_qp_data):

    P, q, A, b, cones, settings = get_info_qp_data
    settings.direct_solve_method = "auto"

    solver = clarabel.DefaultSolver(P, q, A, b, cones, settings)
    solution = solver.solve()
    info = solver.get_info()

    assert solution.status == clarabel.SolverStatus.Solved
    assert info.linsolver.name == "qdldl"
    assert info.linsolver.threads == 1
    assert info.linsolver.direct
    # nonzeros in upper triangle of KKT matrix
    assert info.linsolver.nnzA == 17
    # nonzeros in KKT L factor
    assert info.linsolver.nnzL == 9
