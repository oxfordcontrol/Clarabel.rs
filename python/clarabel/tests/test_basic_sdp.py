import clarabel
import pytest
import numpy as np
from scipy import sparse
from scipy.sparse import vstack


@pytest.fixture
def basic_sdp_data():

    P = sparse.eye(6).tocsc()
    A = sparse.eye(6).tocsc()

    q = np.zeros(6)
    b = np.array([-3., 1., 4., 1., 2., 5.])

    cones = [clarabel.PSDTriangleConeT(3)]
    settings = clarabel.DefaultSettings()
    return P, q, A, b, cones, settings


@pytest.fixture
def basic_sdp_solution():

    refsol = np.array([
        -3.0729833267361095,
        0.3696004167288786,
        -0.022226685581313674,
        0.31441213129613066,
        -0.026739700851545107,
        -0.016084530571308823,
    ])
    refobj = 4.840076866013861

    return refsol, refobj


def test_sdp_feasible(basic_sdp_data, basic_sdp_solution):

    P, q, A, b, cones, settings = basic_sdp_data
    refsol, refobj = basic_sdp_solution

    solver = clarabel.DefaultSolver(P, q, A, b, cones, settings)
    solution = solver.solve()

    assert solution.status == clarabel.SolverStatus.Solved
    assert np.allclose(solution.x, refsol)
    assert np.allclose(solution.obj_val, refobj)
    assert np.allclose(solution.obj_val_dual, refobj)


def test_sdp_empty_cone(basic_sdp_data, basic_sdp_solution):

    P, q, A, b, cones, settings = basic_sdp_data
    refsol, refobj = basic_sdp_solution

    cones = np.append(cones, clarabel.PSDTriangleConeT(0))

    solver = clarabel.DefaultSolver(P, q, A, b, cones, settings)
    solution = solver.solve()

    assert solution.status == clarabel.SolverStatus.Solved
    assert np.allclose(solution.x, refsol)
    assert np.allclose(solution.obj_val, refobj)
    assert np.allclose(solution.obj_val_dual, refobj)


def test_sdp_primal_infeasible(basic_sdp_data):

    P, q, A, b, cones, settings = basic_sdp_data

    A = vstack((A, -A))
    b = np.pad(b, (0, len(b)))
    cones = np.concatenate((cones, cones))

    solver = clarabel.DefaultSolver(P, q, A, b, cones, settings)
    solution = solver.solve()

    assert solution.status == clarabel.SolverStatus.PrimalInfeasible
    assert np.isnan(solution.obj_val)
    assert np.isnan(solution.obj_val_dual)
