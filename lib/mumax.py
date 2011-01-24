#  This file is part of MuMax, a high-performance micromagnetic simulator.
#  Copyright 2011  Arne Vansteenkiste.
#  Use of this source code is governed by the GNU General Public License version 3
#  (as published by the Free Software Foundation) that can be found in the license.txt file.
#  Note that you are welcome to modify this code under the condition that you do not remove any 
#  copyright notices and prominently state that you modified it, giving a relevant date.

from sys import stdin

# INTERNAL
def wait():
	while 1:
		stdin.readlines()

# INTERNAL. Shorthand for running a command with one argument
def do0(command):
	print(command)

# INTERNAL. Shorthand for running a command with one argument
def do1(command, arg):
	print(command + " " + str(arg))

# INTERNAL. Shorthand for running a command with two arguments
def do2(command, arg1, arg2):
	print(command + " " + str(arg1) + " " + str(arg2))

# INTERNAL. Shorthand for running a command with three arguments
def do3(command, arg1, arg2, arg3):
	print(command + " " + str(arg1) + " " + str(arg2) + " " + str(arg3))

# INTERNAL. Shorthand for running a command with arguments
def do(command, args):
	for a in args:
		command += " " + str(a)
	print(command)
		

# Material parameters

# Sets the saturation magnetization in A/m
def msat(m):
	do1("msat", m)

# Sets the exchange constant in J/m
def aexch(a):
	do1("aexch", a)

# Sets the damping parameter
def alpha(a):
	do1("alpha", a)


# Geometry

# Sets the number of FD cells
def gridsize(nx, ny, nz):
	do3("gridsize", nx, ny, nz)

# Sets the size of the magnet, in meters
def partsize(x, y, z):
	do3("partsize", x, y, z)

# Sets the cell size, in meters
def cellsize(x, y, z):
	do3("cellsize", x, y, z)


# Initial magnetization

# Loads the magnetization state from a .omf file
def loadm(filename):
	do1("loadm", filename)

# Sets the magnetization to the uniform state (mx, my, mz)
def uniform(mx, my, mz):
	do3("uniform", mx, my, mz)

# Sets the magnetization in cell with index i,j,k to (mx, my, mz)
def setmcell(i, j, k, mx, my, mz):
	do("setmcell", [i, j, k, mx, my, mz])

# Sets the magnetization in cell position x, y, z (in meters) to (mx, my, mz)
def setm(x, y, z, mx, my, mz):
	do("setmcell", [x, y, z, mx, my, mz])

# Sets the magnetization to a random state
def setrandom():
	do0("setrandom")

# Sets the random number seed
def seed(s):
	do1("seed", s)

# Output

# Single-time save with automatic file name
def save(what, format):
	do2("save", what, format)

# Periodic auto-save
def autosave(what, format, periodicity):
	do3("autosave", what, format, periodicity)


# Solver

# Sets the solver type. E.g.: rk32, rk4, semianal...
def solvertype(solver):
	do1("solvertype", solver)

# Sets the maximum tolerable estimated error per solver step
def maxerror(error):
	do1("maxerror", error)

# Sets the maximum time step 
def maxdt(dt):
	do1("maxdt", dt)

# Sets the minimum time step 
def mindt(dt):
	do1("mindt", dt)

# Sets the maximum magnetization step 
def maxdm(dm):
	do1("maxdm", dm)

# Sets the minimum magnetization step 
def mindm(dm):
	do1("mindm", dm)

# Excitation

# Apply a static field
def staticfield(bx, by, bz):
	do3("staticfield", bx, by, bz)

# Apply an RF field
def rffield(bx, by, bz, freq):
	do("rffield", [bx, by, bz, freq])

# Apply a sawtooth field
def sawtoothfield(bx, by, bz, freq):
	do("sawtoothfield", [bx, by, bz, freq])

# Run

# Relaxes the magnetization up to the specified maximum residual torque
def relax(torque):
	do1("relax", torque)

# Runs for the time specified in seconds
def run(time):
	do1("run", time)


# Misc

# Adds a description tag
def desc(key, value):
	do2("desc", key, value)	

# Save benchmark info to file
def savebenchmark(file):
	do1("savebenchmark", file)

