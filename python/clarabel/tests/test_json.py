import clarabel


def test_json_solve():

    # load and solve with settings in file
    file = "../../examples/data/hs35.json"
    solver = clarabel.read_from_file(file)
    solution = solver.solve()
    assert solution.status == clarabel.SolverStatus.Solved

    # load and solve with custom settings
    settings = clarabel.DefaultSettings()
    settings.max_iter = 1
    solver = clarabel.read_from_file(file, settings)
    solution = solver.solve()
    assert solution.status == clarabel.SolverStatus.MaxIterations
