from mumax import *


solvertype('rk12')
maxerror(1e-4)

# material
msat(800e3)   
aexch(1.3e-11)     


# geometry 
nx = 128
ny = 128
gridsize(nx, ny, 1)    
d=500e-9
partsize(d, d, 25e-9)
# initial magnetization
vortex(1, 1)


# run

tabulate('minmaxmz', True)
autosave("m", "png", 20e-12)
autosave('table', 'ascii', 10e-12)
alpha(2)
run(1e-9)
alpha(0.05)

temperature(400)
for B in range(0, 1000, 10):
	applystatic('field', 0, 0, -B*1E-3)
	run(1e-9)




