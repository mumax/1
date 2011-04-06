#  This file is part of MuMax, a high-performance micromagnetic simulator.
#  Copyright 2011  Arne Vansteenkiste, Ben Van de Wiele.
#  Use of this source code is governed by the GNU General Public License version 3
#  (as published by the Free Software Foundation) that can be found in the license.txt file.
#  Note that you are welcome to modify this code under the condition that you send not remove any 
#  copyright notices and prominently state that you modified it, giving a relevant date.

# @author Arne Vansteenkiste, Ben Van de Wiele

## @package mumax
# MuMax Python API documentation.


from sys import stdin
from sys import stderr
from sys import stdout

## Infinity
inf = float("inf")

# Material parameters

## Sets the saturation magnetization in A/m
def msat(m):
	send1("msat", m)

## Sets the exchange constant in J/m
def aexch(a):
	send1("aexch", a)

## Sets the damping parameter
def alpha(a):
	send1("alpha", a)


## Sets the anisotropy constant K1.
def k1(k):
	send1("k1", k)

## Defines the uniaxial anisotropy axis.
def anisuniaxial(ux, uy, uz):
	send3("anisuniaxial", ux, uy, uz)


## Defines the spin polarization for spin-transfer torque
def spinpolarization(p):
	send1("spinpolarization", p)

## Defines the non-adiabaticity for spin-transfer torque
def xi(xi):
	send1("xi", xi)

## Sets the temperature in Kelvin
def temperature(T):
	send1("temperature", T)


# Geometry

## Sets the number of FD cells
def gridsize(nx, ny, nz):
	send3("gridsize", nx, ny, nz)

## Sets the size of the magnet, in meters
def partsize(x, y, z):
	send3("partsize", x, y, z)

## Sets the cell size, in meters
def cellsize(x, y, z):
	send3("cellsize", x, y, z)

## Sets the maximum cell size, in meters
def maxcellsize(x, y, z):
	send3("maxcellsize", x, y, z)

## Sets periodic boundary conditions.
# The magnetic is repeated nx, ny, nz times in the x,y,z direction (to the left and to the rigth), respectively.
# A value of 0 means no periodicity in that direction.
def periodic(nx, ny, nz):
	send3('periodic', nx, ny, nz)

## Make the geometry an ellipsoid with specified semi-axes.
# Use inf to make it a cyliner along that direction.
def ellipsoid(rx, ry, rz):
	send3("ellipsoid", rx, ry, rz)

# Sets number of periods in given direction
def periodic(Nx, Ny, Nz):
  send3("periodic", Nx, Ny, Nz)


# Initial magnetization

## Loads the magnetization state from a .omf file
def loadm(filename):
	send1("loadm", filename)

## Sets the magnetization to the uniform state (mx, my, mz)
def uniform(mx, my, mz):
	send3("uniform", mx, my, mz)

## Adds random noise to the magnetization
def addnoise(amplitude):
	send1("addnoise", amplitude)

## Initializes the magnetization to a random state
def setrandom():
	send0("setrandom")

## Sets the magnetization to a vortex state
def vortex(circulation, polarization):
	send2("vortex", circulation, polarization)
	
def SBW():
  send0("SBW")

def SNW():
  send0("SNW")

def ABW():
  send0("ABW")

def ANW():
  send0("ANW")

## Sets the magnetization in cell with index i,j,k to (mx, my, mz)
def setmcell(i, j, k, mx, my, mz):
  send_3ints_3floats("setmcell", i, j, k, mx, my, mz)

## Like setmcell but for a range of cells between x1,y1,z1 (inclusive) and x2,y2,z2 (exclusive)
def setmrange(x1, y1, z1, x2, y2, z2, mx, my, mz):
	send("setmrange", [x1, y1, z1, x2, y2, z2, mx, my, mz])

## Sets the magnetization in cell position x, y, z (in meters) to (mx, my, mz)
def setm(x, y, z, mx, my, mz):
	send("setm", [x, y, z, mx, my, mz])

## Sets the magnetization to a ransendm state
def setransendm():
	send0("setransendm")

## Sets the ransendm number seed
def seed(s):
	send1("seed", s)

# Output

## Single-time save with automatic file name
# Format = text | binary
def save(what, format):
	send2("save", what, format)

## Single save of the magnetization to a specified file (.omf)
def savem(filename, format):
	send2("savem", filename, format)

## Single save of the effective field to a specified file (.omf)
def saveh(filename, format):
	send2("saveh", filename, format)

## Periodic auto-save
def autosave(what, format, periodicity):
	send3("autosave", what, format, periodicity)

## Determine what should be saved in the datatable
# E.g.: autosave('m', True)
def tabulate(what, want):
	send2("tabulate", what, want)


# Solver

## Overrides the solver type. E.g.: rk32, rk4, semianal...
def solvertype(solver):
	send1("solvertype", solver)

## Sets the maximum tolerable estimated error per solver step
def maxerror(error):
	send1("maxerror", error)

## Sets the maximum time step in seconds
def maxdt(dt):
	send1("maxdt", dt)

## Sets the minimum time step in seconds
def mindt(dt):
	send1("mindt", dt)

## Sets the maximum magnetization step 
def maxdm(dm):
	send1("maxdm", dm)

## Sets the minimum magnetization step 
def mindm(dm):
	send1("mindm", dm)

# Excitation


def applyfunction(what, func, duration, timestep):
	t=0
	while t<=duration:
		bx,by,bz = func(t)
		applypointwise(what, t, bx, by, bz)
		t+=timestep

## Apply a pointwise-defined field/current defined by a number of time + field points
# The field is linearly interpolated between the defined points
# E.g.:
#  applypointwise('field', 0,       0,0,0) 
#  applypointwise('field', 1e-9, 1e-3,0,0) 
# Sets up a linear ramp in 1ns form 0 to 1mT along X.
# Arbitrary functions can be well approximated by specifying a large number of time+field combinations.
def applypointwise(what, time, bx, by, bz):
	send("applypointwise", [what, time, bx, by, bz])

## Apply a static field/current
def applystatic(what, bx, by, bz):
	send("applystatic", [what, bx, by, bz])

## Apply an RF field/current
def applyrf(what, bx, by, bz, freq):
	send("applyrf", [what, bx, by, bz, freq])

## Apply an RF field/current, slowly ramped in
def applyrframp(what, bx, by, bz, freq, ramptime):
	send("applyrframp", [what, bx, by, bz, freq, ramptime])

## Apply a rotating field/current
def applyrotating(what, bx, by, bz, freq, phaseX, phaseY, phaseZ):
	send("applyrotating", [what, bx, by, bz, freq, phaseX, phaseY, phaseZ])

## Apply a pulsed field/current
def applypulse(what, bx, by, bz, risetime):
	send("applypulse", [what, bx, by, bz, risetime])

## Apply a sawtooth field/current
def applysawtooth(what, bx, by, bz, freq):
	send("applysawtooth", [what, bx, by, bz, freq])

## Apply a rotating RF burst field/current
def applyrotatingburst(what, b, freq, phase, risetime, duration):
	send("applyrotatingburst", [what, b, freq, phase, risetime, duration])

# Run

## Relaxes the magnetization up to the specified maximum residual torque
def relax():
	send0("relax")

## Runs for the time specified in seconds
def run(time):
	send1("run", time)

## Takes one time step
def step():
	send0("step")

## Takes n time steps
def steps(n):
	send1("steps", n)


# Misc

## Adds a description tag
def desc(key, value):
	send2("desc", key, value)	

## Save benchmark info to file
def savebenchmark(file):
	send1("savebenchmark", file)


# Recieve feedback from mumax

## Retrieves an average magnetization component (0=x, 1=y, 2=z).
def getm(component):
	send1("getm", component)
	return recv()

## Retrieves the maximum value of a magnetization component (0=x, 1=y, 2=z).
def getmaxm(component):
	send1("getmaxm", component)
	return recv()

## Retrives the vortex core position in meters, center = 0,0
def getcorepos():
	send0("getcorepos")
	return recv(), recv()

## Retrieves the minimum value of a magnetization component (0=x, 1=y, 2=z).
def getminm(component):
	send1("getminm", component)
	return recv()

## Retrieves the maximum torque in units gamma*Msat
def getmaxtorque(component):
	send1("getmaxtorque")
	return recv()

# Retrieves the total energy in SI units
def getE():
  send0("getE")
  return recv()

## Debug and fine-tuning

## Override whether the exchange interaction is included in the magnetostatic convolution.
def exchinconv(b):
	send1("exchinconv", b)

## Set the exchange type (number of neighbors)
def exchtype(t):
	send1("exchtype", t)

## Override the subcommand for calculating the magnetostatic kernel
def kerneltype(cmd):
	send1("kerneltype", cmd)

## Override whether or not (true/false) the magnetostatic field should be calculated
def demag(b):
	send1("demag", cmd)

# Override whether or not (true/false) the energy should be calculated
def energy(b):
	send1("energy", b)


## @internal
def recv():
	#stderr.write("py_recv: ") #debug
	data = stdin.readline()
	while len(data) == 0 or data[0] != "%":	# skip lines not starting with the % prefix
		stderr.write("py recv():" + data + "\n") #debug
		data = stdin.readline()
	#stderr.write(data + "\n") #debug
	return float(data[1:])

## @internal: version of print() that flushes (critical to avoid communication deadlock)
def myprint(x):
	#stderr.write("py_send: " + str(x) + "\n") #debug
	#stderr.flush()
	stdout.write(x)
	stdout.write("\n")
	stdout.flush()

## @internal. Shorthand for running a command with one argument
def send0(command):
	myprint(command)

## @internal. Shorthand for running a command with one argument
def send1(command, arg):
	myprint(command + " " + str(arg))

## @internal. Shorthand for running a command with two arguments
def send2(command, arg1, arg2):
	myprint(command + " " + str(arg1) + " " + str(arg2))

## @internal. Shorthand for running a command with three arguments
def send3(command, arg1, arg2, arg3):
	myprint(command + " " + str(arg1) + " " + str(arg2) + " " + str(arg3))

## @internal. Shorthand for running a command with arguments
def send(command, args):
	for a in args:
		command += " " + str(a)
	myprint(command)
		
