# Micromagnetic standard problem 4
from mumax import *


# material
msat(800e3)   
aexch(1.3e-11)     
alpha(2)


# geometry 
nx = 64
ny = 64
gridsize(nx, ny, 1)    
partsize(500e-9, 500e-9, 50e-9)

# initial magnetization
uniform(1, 1, 0)


# run
autosave("table", "ascii", 1e-12)
applypointwise('field', 1e-9, 1e-3, 0, 0)
applypointwise('field', 2e-9, 0e-3, 0, 0)
applypointwise('field', 3e-9, 1e-3, 0, 0)
applypointwise('field', 4e-9, 0e-3, 0, 0)
run(5e-9)



