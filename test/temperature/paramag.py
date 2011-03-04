from mumax import *


solvertype('rk12')
mindt(1e-13)
maxdt(1e-13)

# material
msat(800e3)   
aexch(1.3e-11)     


# geometry 
nx = 32
ny = 32
gridsize(nx, ny, 1)    
partsize(96e-9, 96e-9, 3e-9)
# initial magnetization
uniform(1, 1, 0)


# run
autosave("table", "ascii", 5e-12)
autosave("m", "png", 10e-12)

alpha(2)
run(1e-9)
alpha(0.05)

for T in range(0, 1000, 100):
	temperature(T)
	run(1e-9)




