from mumax import *

tabulate('E', True)
msat(800E3)
aexch(1.3E-11)
alpha(2)
anisuniaxial(1, 0, 0)
k1(500)	
gridsize(256, 128, 1)   
partsize(1000e-9, 500e-9, 40e-9)

vortex(1,1)
addnoise(0.1)
alpha(2)
relax()
save("m", "png")

autosave('table', 'ascii', 1e-12)
for i in range(0,300):
	if i > 100:
		i = 100-i
	applystatic('field', i*1.0e-3, 0, 0)
	relax()
	save("m", "text")
	save("m", "png")

