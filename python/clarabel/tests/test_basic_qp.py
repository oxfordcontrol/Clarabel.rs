import clarabel
import pytest
import numpy as np
from scipy import sparse


@pytest.fixture
def basic_qp_data():

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
def basic_qp_data_dual_inf():

    P = sparse.csc_matrix([[1., 1.], [1., 1.]])
    P = sparse.triu(P).tocsc()

    A = sparse.csc_matrix(
        [[1.,   1.],
         [1.,   0.]])

    q = np.array([1., -1.])
    b = np.array([1., 1.])

    cones = [clarabel.NonnegativeConeT(2)]
    settings = clarabel.DefaultSettings()
    return P, q, A, b, cones, settings


def test_qp_feasible(basic_qp_data):

    P, q, A, b, cones, settings = basic_qp_data

    solver = clarabel.DefaultSolver(P, q, A, b, cones, settings)
    solution = solver.solve()

    refsol = np.array([0.3, 0.7])
    refobj = 1.8800000298331538

    assert solution.status == clarabel.SolverStatus.Solved
    assert np.allclose(solution.x, refsol)
    assert np.allclose(solution.obj_val, refobj)
    assert np.allclose(solution.obj_val_dual, refobj)


def test_qp_primal_infeasible(basic_qp_data):

    P, q, A, b, cones, settings = basic_qp_data
    b[0] = -1.
    b[3] = -1.

    solver = clarabel.DefaultSolver(P, q, A, b, cones, settings)
    solution = solver.solve()

    assert solution.status == clarabel.SolverStatus.PrimalInfeasible
    assert np.isnan(solution.obj_val)
    assert np.isnan(solution.obj_val_dual)


def test_qp_dual_infeasible(basic_qp_data_dual_inf):

    P, q, A, b, cones, settings = basic_qp_data_dual_inf

    solver = clarabel.DefaultSolver(P, q, A, b, cones, settings)
    solution = solver.solve()

    assert solution.status == clarabel.SolverStatus.DualInfeasible
    assert np.isnan(solution.obj_val)
    assert np.isnan(solution.obj_val_dual)
