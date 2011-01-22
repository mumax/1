from mumax import *

desc("description", "standard_problem_4")

# material
msat(800e3)   
aexch(1.3e-11)     
alpha(0.02)


# geometry 
nx = 128
ny = 32
gridsize(nx, ny, 1)    

sizex = 500e-9
sizey = 125e-9
sizez = 3e-9
partsize(sizex, sizey, sizez)

# initial magnetization
for i in range(0, nx):
	for j in range(0, ny):
		setmcell(i, j, 0, 1, 0, 0)

# run
autosave("m", "omf", 10e-12)
autosave("m", "png", 10e-12)
autosave("table", "ascii", 10e-12)
run(1e-9)

savebenchmark("benchmark.txt")
