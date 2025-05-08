import clarabel
import pytest
import numpy as np
from scipy import sparse


@pytest.fixture
def get_settings_qp_data():

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
    return P, q, A, b, cones


def test_settings(get_settings_qp_data):

    P, q, A, b, cones = get_settings_qp_data
    settings = clarabel.DefaultSettings()

    solver = clarabel.DefaultSolver(P, q, A, b, cones, settings)
    solver.solve()

    # make and apply bad settings
    settings = clarabel.DefaultSettings()
    settings.direct_solve_method = "foo"
    with pytest.raises(Exception) as e:
        clarabel.DefaultSolver(P, q, A, b, cones, settings)

    print(e)

    # make and apply good settings, then overwrite with bad ones
    settings = clarabel.DefaultSettings()
    settings.presolve_enable = True
    solver = clarabel.DefaultSolver(P, q, A, b, cones, settings)
    solver.solve()

    settings.presolve_enable = False
    with pytest.raises(Exception) as e:
        solver.update(settings=settings)

    print(e)
