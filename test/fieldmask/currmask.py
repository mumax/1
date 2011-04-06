from mumax import *

tabulate('E', True)

# material
msat(800e3)   
aexch(1.3e-11)     
alpha(0.02)

# geometry 
nx = 128
ny = 128
gridsize(nx, ny, 1)    
partsize(500e-9, 500e-9, 3e-9)

# initial magnetization
uniform(1, 1, 0)
alpha(2)
run(5e-9) # relax
alpha(0.001)
save("m", "text")


# run
autosave("m", "omf", 10e-12)
autosave('table', 'ascii', 5e-12)
xi(0.2)
currentmask("mask.omf")
applystatic('j', 1e13, 0, 0)
run(10e-9)


