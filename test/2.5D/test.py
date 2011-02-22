from mumax import *


# material
msat(800e3)   
aexch(1.3e-11)     
alpha(2)


# geometry 
nx = 128
ny = 32
gridsize(nx, ny, 1)    
partsize(500e-9, 125e-9, inf)

# initial magnetization
setrandom()

# run
autosave("m", "text", 100e-12)
autosave("table", "ascii", 100e-12)
run(10e-9)



