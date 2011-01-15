import subprocess
process = subprocess.Popen(["cat"],  stdin=subprocess.PIPE)

def do(command):
	process.stdin.write(command)

do("exit")


