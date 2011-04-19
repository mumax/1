from mumax import *


solvertype('rk12')
maxerror(1e-2)

# material
msat(800e3)   
aexch(1.3e-11)     


# geometry 
nx = 32
ny = 32
gridsize(nx, ny, 1)    
d=96e-9
partsize(d, d, 3e-9)
ellipsoid(d/2., d/2., inf)
# initial magnetization
uniform(1, 1, 0)


# run

autosave("m", "png", 20e-12)
alpha(2)
run(1e-9)
alpha(0.05)

autosave("table", "ascii", 10e-12)
autosave("m", "omf", 20e-12)
temperature(400)
run(100e-9)




