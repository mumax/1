from mumax import *
from math import *

# material
msat(800e3)   
aexch(1.3e-11)     
alpha(0.02)

# geometry 
nx = 512
ny = 512
gridsize(nx, ny, 1)    
partsize(1500e-9, 1500e-9, 3e-9)

# initial magnetization
uniform(1, 1, 0)
alpha(2)
run(5e-9) # relax
alpha(0.01)
save("m", "text")


# run
autosave("m", "omf", 10e-12)
autosave('table', 'ascii', 5e-12)
fieldmask("mask2.omf")

f = 5e9
omega = 2*pi*f
def myfield(t):
	a = t/2e-9
	if a>1:
		a=1
	return a*sin(omega*t), a*cos(omega*t), 0

applyfunction('field', myfield, 10e-9, 10e-12)
run(10e-9)



# End: Data Text
# End: Segment
