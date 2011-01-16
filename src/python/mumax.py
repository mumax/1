#  Copyright 2011  Arne Vansteenkiste
#  Use of this source code is governed by the GNU General Public License version 3
#  (as published by the Free Software Foundation) that can be found in the license.txt file.
#  Note that you are welcome to modify this code under the condition that you do not remove any 
#  copyright notices and prominently state that you modified it, giving a relevant date.

import subprocess
import sys

process = 0

def init(outfile):
	global process
	process = subprocess.Popen(["mumax", "--stdin", outfile],  stdin=subprocess.PIPE)

def do(command):
	global process	
	if process == 0:
		sys.exit("Must call init(out_file) first.")		
	process.stdin.write(command + "\n")

def msat(m):
	do("msat " + str(m))

