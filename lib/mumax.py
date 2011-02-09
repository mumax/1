#  This file is part of MuMax, a high-performance micromagnetic simulator.
#  Copyright 2011  Arne Vansteenkiste.
#  Use of this source code is governed by the GNU General Public License version 3
#  (as published by the Free Software Foundation) that can be found in the license.txt file.
#  Note that you are welcome to modify this code under the condition that you send not remove any 
#  copyright notices and prominently state that you modified it, giving a relevant date.

from sys import stdin
from sys import stderr
from sys import stdout

inf = float("inf")

# INTERNAL
def recv():
	#stderr.write("py_recv: ") #debug
	data = stdin.readline()
	while len(data) == 0 or data[0] != "%":	# skip lines not starting with the % prefix
		data = stdin.readline()
	#stderr.write(data + "\n") #debug
	return float(data[1:])

# INTERNAL: version of print() that flushes (critical to avoid communication deadlock)
def myprint(x):
	#stderr.write("py_send: " + str(x) + "\n") #debug
	#stderr.flush()
	stdout.write(x)
	stdout.write("\n")
	stdout.flush()

# INTERNAL. Shorthand for running a command with one argument
def send0(command):
	myprint(command)

# INTERNAL. Shorthand for running a command with one argument
def send1(command, arg):
	myprint(command + " " + str(arg))

# INTERNAL. Shorthand for running a command with two arguments
def send2(command, arg1, arg2):
	myprint(command + " " + str(arg1) + " " + str(arg2))

# INTERNAL. Shorthand for running a command with three arguments
def send3(command, arg1, arg2, arg3):
	myprint(command + " " + str(arg1) + " " + str(arg2) + " " + str(arg3))

# INTERNAL. Shorthand for running a command with arguments
def send(command, args):
	for a in args:
		command += " " + str(a)
	myprint(command)
		

# Material parameters

# Sets the saturation magnetization in A/m
def msat(m):
	send1("msat", m)

# Sets the exchange constant in J/m
def aexch(a):
	send1("aexch", a)

# Sets the damping parameter
def alpha(a):
	send1("alpha", a)


# Sets the anisotropy constant K1
def k1(k):
	send1("k1", k)

# Defines the uniaxial anisotropy axis.
def anisuniaxial(ux, uy, uz):
	send3("anisuniaxial", ux, uy, uz)


# Geometry

# Sets the number of FD cells
def gridsize(nx, ny, nz):
	send3("gridsize", nx, ny, nz)

# Sets the size of the magnet, in meters
def partsize(x, y, z):
	send3("partsize", x, y, z)

# Sets the cell size, in meters
def cellsize(x, y, z):
	send3("cellsize", x, y, z)

# Sets the maximum cell size, in meters
def maxcellsize(x, y, z):
	send3("maxcellsize", x, y, z)


# Initial magnetization

# Loads the magnetization state from a .omf file
def loadm(filename):
	send1("loadm", filename)

# Sets the magnetization to the uniform state (mx, my, mz)
def uniform(mx, my, mz):
	send3("uniform", mx, my, mz)

# Adds ransendm noise to the magnetization
def addnoise(amplitude):
	send1("addnoise", amplitude)

# Sets the magnetization to a vortex state
def vortex(circulation, polarization):
	send2("vortex", circulation, polarization)

# Sets the magnetization in cell with index i,j,k to (mx, my, mz)
def setmcell(i, j, k, mx, my, mz):
	send("setmcell", [i, j, k, mx, my, mz])

# Sets the magnetization in cell position x, y, z (in meters) to (mx, my, mz)
def setm(x, y, z, mx, my, mz):
	send("setmcell", [x, y, z, mx, my, mz])

# Sets the magnetization to a ransendm state
def setransendm():
	send0("setransendm")

# Sets the ransendm number seed
def seed(s):
	send1("seed", s)

# Output

# Single-time save with automatic file name
def save(what, format):
	send2("save", what, format)

# Periodic auto-save
def autosave(what, format, periodicity):
	send3("autosave", what, format, periodicity)


# Solver

# Sets the solver type. E.g.: rk32, rk4, semianal...
def solvertype(solver):
	send1("solvertype", solver)

# Sets the maximum tolerable estimated error per solver step
def maxerror(error):
	send1("maxerror", error)

# Sets the maximum time step 
def maxdt(dt):
	send1("maxdt", dt)

# Sets the minimum time step 
def mindt(dt):
	send1("mindt", dt)

# Sets the maximum magnetization step 
def maxdm(dm):
	send1("maxdm", dm)

# Sets the minimum magnetization step 
def mindm(dm):
	send1("mindm", dm)

# Excitation

# Apply a static field
def staticfield(bx, by, bz):
	send3("staticfield", bx, by, bz)

# Apply an RF field
def rffield(bx, by, bz, freq):
	send("rffield", [bx, by, bz, freq])

# Apply a sawtooth field
def sawtoothfield(bx, by, bz, freq):
	send("sawtoothfield", [bx, by, bz, freq])

# Run

# Relaxes the magnetization up to the specified maximum residual torque
def relax():
	send0("relax")

# Runs for the time specified in seconds
def run(time):
	send1("run", time)


# Misc

# Adds a description tag
def desc(key, value):
	send2("desc", key, value)	

# Save benchmark info to file
def savebenchmark(file):
	send1("savebenchmark", file)


# Recieve feedback from mumax

# Retrieves an average magnetization component (0=x, 1=y, 2=z).
def getm(component):
	send1("getm", component)
	return recv()
