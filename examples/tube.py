from mumax import *

# example of a hollow nanotube with two domains

solvertype('rk23') # fast but less robust solver, remove this line if the simulation crashes

# material
msat(800e3)
aexch(13e-12)
alpha(10)

# geometry
Nx = 768 # 16 * 2^n * (1,3,5 or 7) is best for performance. This is 3*256
Ny = 16
gridSize(Nx, Ny, Ny)

length = 1500e-9
diam = 32e-9
partSize(length, diam, diam)

# define a tube
r1 = Ny/2 #tube outer radius (cells)
r2 = 4 # tube inner radius (cells)

for i in range(Ny):
	for j in range(Ny):
		# standard way to get distance from tube axis. +0.5: look at center of cell for symmetry
		y = (i+0.5)-Ny/2
		z = (j+0.5)-Ny/2
		r = y*y + z*z
		if r > r1*r1 or r < r2*r2:
			setMsatRange(0,i,j,	Nx-1,i,j,		0)


# inital magnetization: vortex wall
setmRange(0,0,0,	Nx/2,15,15,	1,0,0) # sets the range x=0..285 y=0..15 z=0..15 (upper bounds inclusive) to m = (1,0,0)
setmRange(Nx/2,0,0,	Nx-1,15,15,	-1,0,0)

# set some initial chirality to get a vortex wall
wallwidth = 4 # cells
setmRange(Nx/2-wallwidth,0,0,	Nx/2+wallwidth,Ny/2,Ny-1,	0,0,1)  
setmRange(Nx/2-wallwidth,Ny/2,0,	Nx/2+wallwidth,Ny-1,Ny-1,	0,0,-1)

# add a little randomness to break symmetry
addNoise(0.01)


# schedule output
autosave('m', 'binary', 100e-12)
autosave('table', 'ascii', 10e-12)

# run!
run(2e-9)

save('m', 'binary')
