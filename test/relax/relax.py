from mumax import *

msat(800E3)
aexch(1.3E-11)
alpha(1)
gridsize(128, 128, 1)   
partsize(256e-9, 256e-9, 40e-9)

setrandom()
tabulate('E', True)
autosave('table', 'ascii', 0.000001e-12)
autosave('m', 'png', 1e-12)
relax()


