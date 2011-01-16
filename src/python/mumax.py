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
	mumax_subprocess = subprocess.Popen(["mumax", "--stdin", outfile],  stdin=subprocess.PIPE)


# Asks mumax to exit and waits for the mumax subprocess to do so.
def exit():
	global mumax_subprocess
	do("exit")
	mumax_subprocess.stdin.close()
	status = mumax_subprocess.wait()
	print("mumax exited with status " + str(status))

# Runs a general mumax command
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
		

# Sets the saturation magnetization in A/m
def msat(m):
	do1("msat", m)

# Sets the exchange constant in J/m
def aexch(a):
	do("aexch " + str(a))

# Sets the number of FD cells
def gridsize(nx, ny, nz):
	do3("gridsize", nx, ny, nz)

# Sets the size of the magnet, in meters
def partsize(x, y, z):
	do3("partsize", x, y, z)

# Sets the magnetization to the uniform state (mx, my, mz)
def uniform(mx, my, mz):
	do3("uniform", mx, my, mz)

# Runs for the time specified in seconds
def run(time):
	print("run!")
	do1("run", time)
