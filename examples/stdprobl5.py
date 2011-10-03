# Spintorque standard problem
from mumax import *

aexch(  13e-12)
msat(   8e5)

partSize(  100e-9,100e-9,10e-9)
gridSize(  64,     64,	 1)

#relax
vortex (1,1)
alpha(1)
run(1e-9)

#run
alpha( 0.1)
currentMask('uniform.omf')
applyStatic("j", 1e12 ,0 ,0 )
autosave("m"     ,"omf"  , 20e-12)
autosave("table" ,"ascii" ,20e-12)
run(10e-9)
