# Micromagnetic standard problem 4
from mumax import *


# material
msat(800e3)   
aexch(1.3e-11)     
alpha(0.02)


# geometry 
nx = 128
ny = 128
gridsize(nx, ny, 1)    
partsize(500e-9, 500e-9, 20e-9)

# initial magnetization
setrandom()
alpha(1)
save("m", "omf")

tabulate('B', False)
tabulate('E', True)

# run
autosave("m", "omf", 100e-12)
autosave("table", "ascii", 1e-12)
run(10e-9)



