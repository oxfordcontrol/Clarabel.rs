import clarabel
import os

thisdir = os.path.dirname(__file__)
filename = os.path.join(thisdir, "../data/hs35.json")
print(filename)

# Load problem data from JSON file
solver = clarabel.read_from_file(filename)
solution = solver.solve()

# export problem data to JSON file
# filename = os.path.join(thisdir, "../data/out.json")
# solver.write_to_file(filename)
