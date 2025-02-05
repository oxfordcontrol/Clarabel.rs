using Clarabel, JuMP, SparseArrays
using ClarabelRs

# try loading a test problem with/without settings 
filename = "../../../../examples/data/hs35.json"

# with settings in file 
solver = ClarabelRs.load_from_file(filename)
solution = ClarabelRs.solve!(solver)
@assert solution.status == Clarabel.SOLVED

# with custom settings
settings = Clarabel.Settings() 
settings.max_iter = 1
solver = ClarabelRs.load_from_file(filename,settings)
solution = ClarabelRs.solve!(solver)
@assert solution.status == Clarabel.MAX_ITERATIONS
