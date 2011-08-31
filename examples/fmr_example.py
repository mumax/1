# This example illustrates how to define an arbitary field
# B = f(t)
# current densities j(t) can be defined similarly.

from mumax import *
from math import sin, pi

# Nickel:
msat(490e3)   
aexch(9e-12)     
alpha(0.01)
k1(-5.7e3)
anisUniaxial(0, 0, 1)

# geometry 
nx = 128
ny = 128
gridSize(nx, ny, 1)    
partSize(300e-9, 300e-9, 50e-9)

# initial magnetization
uniform(1, 0, 0)

# custom-defined RF field along Y-axis
A = 1e-3    	# RF amplitude (T)
def myfield(t):
	f = 3000e6  # frequency = 3GHz
	by = A * sin( 2*pi * f * t)
	return 0, by, 0

# apply the custum defined field "myfield" for 2ns, sample its value every 10ps
# (linear interpolation between samples)
applyFunction('field', myfield, 2e-9, 10e-12) 

# current densities j(t) (in A/m^2) can be defined similarly:
#currentMask("mask.omf")
applyStatic('j', 10e9, 0, 0) 

# schedule output and run
autosave('table', 'ascii', 20e-12)
autosave('m', 'binary', 50e-12)
run(2e-9) # run for 2ns

