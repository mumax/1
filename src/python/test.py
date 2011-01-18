from mumax import *

init("test.out")
msat(750e3)
aexch(1.3e-11)
gridsize(64, 64, 1)
partsize(500e-9, 500e-9, 50e-9)
uniform(1, 0, 0)
run(1e-9)
exit()
