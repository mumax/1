# Micromagnetic standard problem 4
from mumax import *
from math import *

desc("description", "standard_problem_4")

# material
msat(800e3)   
aexch(1.3e-11)     
alpha(0.02)


# geometry 
nx = 32
ny = 32
gridsize(nx, ny, 1)    
partsize(125e-9, 125e-9, 3e-9)

# initial magnetization
uniform(1, 1, 0)
alpha(2)


# run
autosave("table", "text", 1e-12)

def myfield(t):
	bx = 1e-3 * sin(2 *pi * 1e9 * t)
	by = 1e-3 * cos(2 *pi * 1e9 * t)
	bz = 0
	return bx, by, bz

applyfunction('field', myfield, 1e-9, 10e-12)
run(1e-9)



