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

