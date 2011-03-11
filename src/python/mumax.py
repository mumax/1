#  This file is part of MuMax, a high-performance micromagnetic simulator.
#  Copyright 2011  Arne Vansteenkiste.
#  Use of this source code is governed by the GNU General Public License version 3
#  (as published by the Free Software Foundation) that can be found in the license.txt file.
#  Note that you are welcome to modify this code under the condition that you do not remove any 
#  copyright notices and prominently state that you modified it, giving a relevant date.

import subprocess
import sys


mumax_subprocess = 0

# Starts the mumax subprocess, saving output in outfile.out
def init(outfile):
	global mumax_subprocess
	mumax_subprocess = subprocess.Popen(["mumax", "--cpu", "--stdin", outfile],  stdin=subprocess.PIPE)


# Asks mumax to exit and waits for the mumax subprocess to do so.
def exit():
	global mumax_subprocess
	do("exit")
	mumax_subprocess.stdin.close()
	status = mumax_subprocess.wait()
	print("mumax exited with status " + str(status))

# Runs a general mumax command passed as a string. E.g.: "msat 800e3"
def do(command):
	global mumax_subprocess	
	if mumax_subprocess == 0:
		sys.exit("Must call init(out_file) first.")		
	mumax_subprocess.stdin.write(command + "\n")

# INTERNAL. Shorthand for running a command with one argument
def do1(command, arg):
	do(command + " " + str(arg))

# INTERNAL. Shorthand for running a command with two arguments
def do2(command, arg1, arg2):
	do(command + " " + str(arg1) + " " + str(arg2))

# INTERNAL. Shorthand for running a command with three arguments
def do3(command, arg1, arg2, arg3):
	do(command + " " + str(arg1) + " " + str(arg2) + " " + str(arg3))

# INTERNAL. Shorthand for running a command with arguments
def doA(command, args):
	for i in args:
		command += " " + str(args[i])
	do(command)
		

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
	
# Sets the number of periodic repetitions
def periodic(Nx, Ny, Nz):
  do3("periodic", Nx, Ny, Nz)


# Initial magnetization

# Loads the magnetization state from a .omf file
def loadm(filename):
	do1("loadm", filename)

# Sets the magnetization to the uniform state (mx, my, mz)
def uniform(mx, my, mz):
	do3("uniform", mx, my, mz)


# Output

# Single-time save with automatic file name
def save(what, format):
	do2("save", what, format)

# Periodic auto-save
def autosave(what, format, periodicity):
	do3("save", what, format, periodicity)


# Excitation

# Apply a static field
def staticfield(bx, by, bz):
	do3("staticfield", bx, by, bz)

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

