from mumax import *

desc("description", "standard_problem_4")

# material
msat(800e3)   
aexch(1.3e-11)     
alpha(2)


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
autosave("m", "binary", 100e-12)
autosave("table", "ascii", 10e-12)
rffield(1, 1, 1, 1e9)
sawtoothfield(1e-3, 0, 0, 1000e6)
run(10e-9)
m = getmPos(0, 100, 10, 0)

savebenchmark("benchmark.txt")
