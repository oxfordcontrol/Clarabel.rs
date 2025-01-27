import clarabel
import pytest
import numpy as np
from scipy import sparse


@pytest.fixture
def updating_test_data():

    P = sparse.csc_matrix([[40000., 1.], [1., 20000.]])
    P = sparse.triu(P).tocsc()

    A = sparse.csc_matrix(
        [[1.,  0.],
         [0.,  1.],
         [-1., 0.],
         [0., -1.]])

    q = np.array([10000., 10000.])
    b = np.array([1., 1., 1., 1.])

    cones = [clarabel.NonnegativeConeT(2), clarabel.NonnegativeConeT(2)]
    settings = clarabel.DefaultSettings()
    return P, q, A, b, cones, settings


def test_update_P_matrix_form(updating_test_data):

    P, q, A, b, cones, settings = updating_test_data

    solver1 = clarabel.DefaultSolver(P, q, A, b, cones, settings)
    solver1.solve()

    # change P and re-solve
    P2 = P.copy()
    P2[0, 0] = 100.

    solver1.update(P=P2)
    solution1 = solver1.solve()

    # new solver
    solver2 = clarabel.DefaultSolver(P2, q, A, b, cones, settings)
    solution2 = solver2.solve()

    assert np.allclose(solution1.x, solution2.x)


def test_update_P_vector_form(updating_test_data):

    P, q, A, b, cones, settings = updating_test_data

    solver1 = clarabel.DefaultSolver(P, q, A, b, cones, settings)
    solver1.solve()

    # change P and re-solve
    P2 = P.copy()
    P2[0, 0] = 100.

    solver1.update(P=P2.data)
    solution1 = solver1.solve()

    # new solver
    solver2 = clarabel.DefaultSolver(P2, q, A, b, cones, settings)
    solution2 = solver2.solve()

    assert np.allclose(solution1.x, solution2.x)


def test_update_P_tuple(updating_test_data):

    P, q, A, b, cones, settings = updating_test_data

    solver1 = clarabel.DefaultSolver(P, q, A, b, cones, settings)
    solver1.solve()

    # change P and re-solve
    values = [3., 5.]
    index = [1, 2]
    Pnew = (index, values)
    solver1.update(P=Pnew)
    solution1 = solver1.solve()

    # new solver
    P00 = P[0, 0]
    P2 = sparse.csc_matrix([[P00, 3.], [0., 5.]]).tocsc()
    solver2 = clarabel.DefaultSolver(P2, q, A, b, cones, settings)
    solution2 = solver2.solve()

    assert np.allclose(solution1.x, solution2.x)


def test_update_A_matrix_form(updating_test_data):

    P, q, A, b, cones, settings = updating_test_data

    solver1 = clarabel.DefaultSolver(P, q, A, b, cones, settings)
    solver1.solve()

    # change A and re-solve
    A2 = A.copy()
    A2.data[2] = -1000.

    solver1.update(A=A2)
    solution1 = solver1.solve()

    # new solver
    solver2 = clarabel.DefaultSolver(P, q, A2, b, cones, settings)
    solution2 = solver2.solve()

    assert np.allclose(solution1.x, solution2.x)


def test_update_A_vector_form(updating_test_data):

    P, q, A, b, cones, settings = updating_test_data

    solver1 = clarabel.DefaultSolver(P, q, A, b, cones, settings)
    solver1.solve()

    # change A and re-solve
    A2 = A.copy()
    A2.data[2] = -1000.

    solver1.update(A=A2.data)
    solution1 = solver1.solve()

    # new solver
    solver2 = clarabel.DefaultSolver(P, q, A2, b, cones, settings)
    solution2 = solver2.solve()

    assert np.allclose(solution1.x, solution2.x)


def test_update_A_tuple(updating_test_data):

    P, q, A, b, cones, settings = updating_test_data

    solver1 = clarabel.DefaultSolver(P, q, A, b, cones, settings)
    solver1.solve()

    # change P and re-solve
    values = [0.5, -0.5]
    index = [1, 2]
    Anew = (index, values)
    solver1.update(A=Anew)
    solution1 = solver1.solve()

    # new solver
    A2 = A.copy()
    A2.data[1] = 0.5
    A2.data[2] = -0.5
    solver2 = clarabel.DefaultSolver(P, q, A2, b, cones, settings)
    solution2 = solver2.solve()

    assert np.allclose(solution1.x, solution2.x)


def test_update_q(updating_test_data):

    P, q, A, b, cones, settings = updating_test_data

    solver1 = clarabel.DefaultSolver(P, q, A, b, cones, settings)
    solver1.solve()

    # change q and re-solve
    q2 = q.copy()
    q2[1] = 10.

    solver1.update(q=q2)
    solution1 = solver1.solve()

    # new solver
    solver2 = clarabel.DefaultSolver(P, q2, A, b, cones, settings)
    solution2 = solver2.solve()

    assert np.allclose(solution1.x, solution2.x)


def test_update_q_tuple(updating_test_data):

    P, q, A, b, cones, settings = updating_test_data

    solver1 = clarabel.DefaultSolver(P, q, A, b, cones, settings)
    solver1.solve()

    # change q and re-solve
    values = [10.]
    index = [1]
    qdata = (index, values)

    solver1.update(q=qdata)
    solution1 = solver1.solve()

    # new solver
    q2 = q.copy()
    q2[1] = 10.
    solver2 = clarabel.DefaultSolver(P, q2, A, b, cones, settings)
    solution2 = solver2.solve()

    assert np.allclose(solution1.x, solution2.x)


def test_update_b(updating_test_data):

    P, q, A, b, cones, settings = updating_test_data

    solver1 = clarabel.DefaultSolver(P, q, A, b, cones, settings)
    solver1.solve()

    # change b and re-solve
    b2 = b.copy()
    b2[0] = 0.

    solver1.update(b=b2)
    solution1 = solver1.solve()

    # new solver
    solver2 = clarabel.DefaultSolver(P, q, A, b2, cones, settings)
    solution2 = solver2.solve()

    assert np.allclose(solution1.x, solution2.x)


def test_update_b_tuple(updating_test_data):

    P, q, A, b, cones, settings = updating_test_data

    solver1 = clarabel.DefaultSolver(P, q, A, b, cones, settings)
    solver1.solve()

    # change b and re-solve
    values = [0., 0.]
    index = [1, 3]
    bdata = (index, values)

    solver1.update(b=bdata)
    solution1 = solver1.solve()

    # new solver
    b2 = np.array([1., 0., 1., 0.])
    solver2 = clarabel.DefaultSolver(P, q, A, b2, cones, settings)
    solution2 = solver2.solve()

    assert np.allclose(solution1.x, solution2.x)


def test_settings(updating_test_data):

    P, q, A, b, cones, settings = updating_test_data

    solver = clarabel.DefaultSolver(P, q, A, b, cones, settings)
    solution = solver.solve()

    assert solution.status == clarabel.SolverStatus.Solved

    # extract settings, modify and reassign
    settings = solver.get_settings()
    settings.max_iter = 1
    solver.update(settings=settings)
    solution = solver.solve()

    assert solution.status == clarabel.SolverStatus.MaxIterations


def test_presolved_update(updating_test_data):

    # check that updates are rejected properly after presolving

    P, q, A, b, cones, settings = updating_test_data

    solver = clarabel.DefaultSolver(P, q, A, b, cones, settings)

    # presolve enabled but nothing eliminated
    assert solver.is_data_update_allowed()

    # presolved disabled in settings
    b[0] = 1e30
    settings.presolve_enable = False
    solver = clarabel.DefaultSolver(P, q, A, b, cones, settings)
    assert solver.is_data_update_allowed()

    # should be eliminated
    b[0] = 1e30
    settings.presolve_enable = True
    solver = clarabel.DefaultSolver(P, q, A, b, cones, settings)
    assert not solver.is_data_update_allowed()
