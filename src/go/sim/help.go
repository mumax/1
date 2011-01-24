//  Copyright 2010  Arne Vansteenkiste
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package sim

import (
	"flag"
	"fmt"
	"os"
	"iotool"
)

// Display help message and generate example input file

func Help() {
	flag.PrintDefaults()
}

// Error message for no input files 
func NoInputFiles() {
	helpmsg := `No input files. Usage: mumax file.in
Type mumax -example=file.in to create an example input file.
mumax -help will print more command line options.
`
	fmt.Fprintln(os.Stderr, helpmsg)
}

// Creates an example file
func Example(file string) {
	out := iotool.MustOpenWRONLY(file)
	defer out.Close()
	out.Write([]byte(EXAMPLE))
	fmt.Println("Created example input file: ", file)
}

// Error message for unknown input file extension
func UnknownFileFormat(extension string) {
	fmt.Fprintln(os.Stderr, "Unknown file format: ", extension)
	fmt.Fprintln(os.Stderr, "Recognized extensions: ", known_extensions)
}

const EXAMPLE = `
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

# Anisotropy type, axes and constant(s)
# anisuniaxial	0 0 1	# axis = Z
# k1			-50e3	# in J/m^3, sign determines hard/easy axis.

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

# randomseed 1
# setrandom # sets the magnetization to a random state
#
# vortex    circ pol  # sets a vortex state with given circulation and polarization, e.g.: vortex 1 -1
#
# Finally, an existing magnetization can be changed as follows:
#
# addnoise  amplitude # adds random noise to break the symmetry,
#                     # e.g.: addnoise 0.01
# setmrange x0 y0 z0 x1 y1 z1 mx my mz  # manually set the magnetization in the specified x-y-z range (of cell indices) to mx, my, mz
#                                       # e.g.: setmrange 5 12 23 7 13 25 1. 0. 0.
#
# setmcell i j k mx my mz # sets the magnetization of cell i j k to mx,my,mz (handy in a python loop, e.g.)
#


# (4) Schedule output
# ___________________

autosave  m     png   50E-12  # magnetization will be saved every 50 ps in PNG format
autosave  m     text   50e-12  # magnetization will be saved every 50 ps in OOMMFs .omf format 
#autosave  m     binary   50e-12  # magnetization will be saved every 50 ps in OOMMFs .omf format 
autosave  table ascii 10E-12  # tabular output will be saved every 10ps in OOMMFs .odt format


# (5) Configure solver
# _______

# All of this is optional:
# solvertype rk23		# chooses the type of solver. Options: rk1, rk12, rk2, rk23, rk3, rk4, rk45
# maxerror 1e-5		# maximum estimated error per step. The real error will be smaller.
# maxdt	1e-12	      # maximum time step in seconds
# mindt  1e-15			# minimum time step in seconds
# maxdm	0.1		   # maximum magnetization change per time step
# mindm	1e-3		   # minimum magnetization change per time step


# (6) Relax
# _______

# To relax the magnetization, set a high alpha temporarily and run for a while.

alpha   2
relax 1e-2	# relax until the residual torque is < 1e-4. A high damping should be set first.
alpha   0.02


# (7) Excite
# __________

# Apply a static magnetic field, expressed in tesla:

staticfield 	-24.6E-3     4.3E-3    0 

# rffield 1e-3 0 0 500e6	# apply a sinusoidal RF field of 1mT along X with a 500MHz frequency
# sawtoothfield 1e-3 0 0 100e6	# apply a sawtooth RF field of 1mT along X with a 100MHz frequency (can be used for hysteresis sweeps)
# pulsedfield 0 1e-3 0 1e-9 1e-12 # apply a pulse of 1mT along Y with 1ns duration, 1ps rise- and fall times (linear edge, 0-100%).

# Apply electrical current:

# xi                  0.05        # Degree of non-adiabaticity
# spinpolarization    0.8         # Degree of spin-polarization
# currentdensity      1e12 0 0    # Curent density in A/m^2

# (8) Run
# ______

# Run for a specified amount of time, in seconds:

run          	1E-9             

`
