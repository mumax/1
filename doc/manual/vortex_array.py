# This example shows how to make a vortex array.  The polarization and circulation of each vortex can be set separately.

from mumax import *

# material
msat(800e3)
aexch(13.0e-12)
alpha(1)

# for periodic boundary conditions in x- and y-dimension
# periodic(20,20,0)


# geometry
Nvrtx_x = 8     #number of vortices in x-direction
Nvrtx_y = 8     #number of vortices in y-direction

thickness = 10e-9
vortex_size_x = 125e-9
vortex_size_y = 100e-9
separation_x = 25e-9
separation_y = 50e-9
cell_size = 3.125e-9

unit_size_x = vortex_size_x + separation_x;
unit_size_y = vortex_size_y + separation_y;

xsize = Nvrtx_x * unit_size_x
ysize = Nvrtx_y * unit_size_y

cellSize( cell_size, cell_size, thickness )
partSize( xsize, ysize, thickness )


# Each vortex in the array is set separately, possibly with different circulation and polarization.
# The vortices have a rectangular shape.
circulation = 1
polarization = 1
for i in range(Nvrtx_x):
  for j in range(Nvrtx_y):
    vortexInArray(i, j, unit_size_x, unit_size_y, separation_x, separation_y, circulation, polarization)

# To set vortices with an ellipsoidal shape, a mask can be put over the vortex array.
dotArrayEllips(unit_size_x, unit_size_y, separation_x, separation_y, Nvrtx_x, Nvrtx_y)

# make the structure relax towards the ground state.
relax()
save("m", "png")
