# Micromagnetic standard problem 4
from mumax import *

# material
msat(800e3)   
aexch(1.3e-11)     
alpha(0.02)


# geometry 
nx = 128
ny = 32
gridSize(nx, ny, 1)    
partSize(500e-9, 125e-9, 3e-9)

# initial magnetization
uniform(1, 1, 0)
alpha(2)
#relax:
run(10E-9)
alpha(0.02)
save("m", "omf")


# run
autosave("m", "omf", 10e-12)
autosave("m", "png", 10e-12)
applyStatic('field', -24.6E-3, 4.3E-3, 0)
run(1e-9)



