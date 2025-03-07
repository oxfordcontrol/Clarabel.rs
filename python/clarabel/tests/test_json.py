import clarabel
import os


def test_json_solve():

    datafile = "hs35.json"

    # load and solve with settings in file
    thisdir = os.path.dirname(__file__)
    datadir = os.path.join(thisdir, "../../../examples/data/")
    file = os.path.join(datadir, datafile)

    solver = clarabel.load_from_file(file)
    solution = solver.solve()
    assert solution.status == clarabel.SolverStatus.Solved

    # load and solve with custom settings
    settings = clarabel.DefaultSettings()
    settings.max_iter = 1
    solver = clarabel.load_from_file(file, settings)
    solution = solver.solve()
    assert solution.status == clarabel.SolverStatus.MaxIterations
