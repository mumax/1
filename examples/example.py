# MuMax example file
#
# The contents of an input file are case-insensitive.
# Lines starting with a '#', like this line, are comments, they are ignored by mumax.
# Semicolons (;) may optionally be used to separate commands or end lines.
#
# The commands in this file are executed in the order they appear in.
# E.g.:
# msat 800e3; run 1e-9
# is not the same as:
# run 1e-9; msat 800e3

#

# (1) Material parameters
# _______________________

# Saturation magnetization in A/m
msat       	800E3

# Exchange constant in J/m
aexch      	1.3E-11

# Damping coefficient
alpha      	0.02



# (2) Geometry
# ____________

# To define the geometry, you must specify exactly 2 out of these 3 statements:
#
# gridsize Nx Ny Nz          # defines the number of FD cells
# partsize SizeX SizeY SizeZ # defines the total magnet size, in meters
# cellsize SizeX SizeY SizeZ # defines the size of each FD cell, in meters
#
# When 2 of these sizes are specified, the 3d one is caluclated automatically.
# It is most usual to define the partsize and gridsize, and have the cellsize be calculated automatically.
#
# Note: the gridsize should preferably contain highly factorable numbers,
# like powers of 2 or small multiples of powers of two.
# E.g.: a 128 x 256 x 4 gridsize will run much faster than a 101 x 223 x 3 grid.
#
# Note: For 2D simulations, use Nx x Ny x 1, not 1 x Nx x Ny
#

gridsize       	128 	  32 	    1    
partsize   	    500E-9  125E-9  3E-9

# To define the shape, use one of the built-in shapes or apply a PNG mask

#mask examplemask.png

# built-in geometries:
# ellipsoid rx ry rz   # ellipsoid with semi-axes rx, ry, rz
#                      # note: use rz = inf to get a cylinder


# (3) Initial magnetization
# _________________________

# There are several commands to set the initial magnetization:
# A previously stored magnetization state can be loaded with "loadm filename":
#
# loadm myfile.tensor
#
# Also, there are a few pre-defined configurations available:
#
# uniform   mx my mz  # sets a uniform state, e.g.: 

uniform 1 0 0

# vortex    circ pol  # sets a vortex state with given circulation and polarization, e.g.: vortex 1 -1
#
# Finally, an existing magnetization can be changed as follows:
#
# addnoise  amplitude # adds random noise to break the symmetry,
#                     # e.g.: addnoise 0.01
# setmrange x0 y0 z0 x1 y1 z1 mx my mz  # manually set the magnetization in the specified x-y-z range (of cell indices) to mx, my, mz
#                                       # e.g.: setmrange 5 12 23 7 13 25 1. 0. 0.

# (4) Schedule output
# ___________________

autosave  m     png   50E-12  # magnetization will be saved every 50 ps in PNG format
autosave  m     ascii 50e-12  # magnetization will be saved every 50 ps in ascii text format
autosave  table ascii 10E-12  # tabular output will be saved every 10ps in ascii text format


# (5) Relax
# _______

# To relax the magnetization, set a high alpha temporarily and run for a while.
# A more user-friendly "relax" function is under development.

alpha   2
run     5e-9    # runs the simulation for 5ns
alpha   0.02


# (6) Excite
# __________

# Apply a static magnetic field, expressed in tesla:

staticfield 	-24.6E-3     4.3E-3    0 

# Apply electrical current:
# Note: This is not yet implemented on the CPU, works only on the GPU

# xi                  0.05        # Degree of non-adiabaticity
# spinpolarization    0.8         # Degree of spin-polarization
# currentdensity      1e12 0 0    # Curent density in A/m^2

# (7) Run
# ______

# Run for a specified amount of time, in seconds:

run          	1E-9             

