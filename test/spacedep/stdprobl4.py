# Micromagnetic standard problem 4
from mumax import *

# material
msat(800e3)   
aexch(1.3e-11)     
alpha(0.02)


# geometry 
nx = 128
ny = 32
gridsize(nx, ny, 1)    
partsize(500e-9, 125e-9, 3e-9)


#setmsat(nx/2, ny/2, 0, 0)

# initial magnetization
uniform(1, 1, 0)
alpha(2)
run(1e-9) # relax
alpha(0.02)
save("m", "omf")


for i in range(0, nx/2):
	for j in range(0, ny):
		setalpha(i, j, 0, 100)
		#setmsat(i, j, 0, 1)

# run
autosave("m", "omf", 10e-12)
autosave('table', 'ascii', 5e-12)
applystatic('field', -24.6E-3, 4.3E-3, 0)
run(1e-9)


