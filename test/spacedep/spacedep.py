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
partsize(500e-9, 500-9, 3e-9)
uniform(1, 1, 0)

for i in range(0, nx):
	for j in range(0, ny):
		setmsat(i, j, 0, 1)
		setalpha(i, j, 0, nx/100.)

# initial magnetization
save("m", "omf")


# run
autosave("m", "omf", 10e-12)
autosave('table', 'ascii', 5e-12)
applystatic('field', -24.6E-3, 4.3E-3, 0)
run(1e-9)


