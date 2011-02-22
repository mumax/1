from mumax import *

kerneltype('mumaxkern-cpu')
#exchtype(0)
#exchinconv(0)

# material
msat(1448e3)
aexch(1.5e-11)     
alpha(1)
anisuniaxial(0, 1, 0)
k1(40e3)

# geometry 
nx = 1024
ny = 128
gridsize(nx, ny, 1)    
partsize(5.12e-6, 0.64e-6, inf)


# initial magnetization
uniform(0, 0.01, 1)

# run
autosave("m", "binary", 100e-12)
autosave("table", "ascii", 10e-12)
maxdt(1e-12)
mindt(1e-15)
mindm(1e-5)
run(10e-9)




