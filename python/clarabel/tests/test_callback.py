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


def test_termination_callback(callback_qp_data):

    P, q, A, b, cones, settings = callback_qp_data

    def test_termination_fcn(info: clarabel.DefaultInfo) -> bool:
        return info.iterations > 2

    # solves
    solver = clarabel.DefaultSolver(P, q, A, b, cones, settings)
    solution = solver.solve()
    assert solution.status == clarabel.SolverStatus.Solved

    # stops early
    solver.set_termination_callback(test_termination_fcn)
    solution = solver.solve()
    assert solution.iterations == 3, "Should stop after 3 iterations"
    assert solution.status == clarabel.SolverStatus.CallbackTerminated

    # works again
    solver.unset_termination_callback()
    solution = solver.solve()
    assert solution.status == clarabel.SolverStatus.Solved



def test_termination_callback_with_state(callback_qp_data):

    P, q, A, b, cones, settings = callback_qp_data

    counter = [-1] # list as a mutable container

    def test_termination_with_state_fcn(info: clarabel.DefaultInfo) -> bool:
        counter[0] += 1
        if counter[0] >= 3:
            return True
        return False

    # solves
    solver = clarabel.DefaultSolver(P, q, A, b, cones, settings)
    solution = solver.solve()
    assert solution.status == clarabel.SolverStatus.Solved

    # stops early
    solver.set_termination_callback(test_termination_with_state_fcn)
    solution = solver.solve()
    assert solution.iterations == 3, "Should stop after 3 iterations"
    assert solution.status == clarabel.SolverStatus.CallbackTerminated

    # works again
    solver.unset_termination_callback()
    solution = solver.solve()
    assert solution.status == clarabel.SolverStatus.Solved