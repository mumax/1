import subprocess

process = 0

def init(outfile):
	global process
	process = subprocess.Popen(["mumax", "--stdin", outfile],  stdin=subprocess.PIPE)

def do(command):
	global process
	process.stdin.write(command)



