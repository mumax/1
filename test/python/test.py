# Micromagnetic standard problem 4
from mumax import *

desc("description", "standard_problem_4")

# material
msat(800e3)   
aexch(1.3e-11)     
alpha(0.02)


# geometry 
nx = 128
ny = 32
gridsize(nx, ny, 1)    
partsize(500e-9, 125e-9, 3e-9)

# initial magnetization
seed(1)
setrandom()


# run
autosave("m", "omf", 10e-12)
autosave("m", "png", 10e-12)
autosave("table", "ascii", 10e-12)
run(1e-9)

savebenchmark("benchmark.txt")
