import clarabel
import pytest
import numpy as np
from scipy import sparse


@pytest.fixture
def test_print_data():

    P = sparse.csc_matrix([[4.]])
    P = sparse.triu(P).tocsc()

    A = sparse.csc_matrix(
        [[1.]])

    q = np.array([1.])
    b = np.array([1.])

    cones = [clarabel.NonnegativeConeT(1)]
    settings = clarabel.DefaultSettings()
    solver = clarabel.DefaultSolver(P, q, A, b, cones, settings)
    return solver


def test_print_to_file(test_print_data):

    import tempfile
    import os

    solver = test_print_data

    with tempfile.NamedTemporaryFile(delete=False) as file:
        filename = file.name

    try:
        solver.print_to_file(filename)
        solver.solve()
        with open(filename, "r") as f:
            contents = f.read()

        assert "Clarabel.rs" in contents, f"Bad file contents: {contents}"

    finally:
        os.remove(filename)


def test_print_to_buffer(test_print_data):

    solver = test_print_data
    buffer = solver.print_to_buffer()
    solver.solve()
    contents = solver.get_print_buffer()

    assert "Clarabel.rs" in contents, f"Bad buffer contents: {buffer}"
