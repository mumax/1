from mumax import *

# we chose lex fixed to 1

msat(800e3)
aexch(1.3e-11)
alpha(1)

#lex = 5.6858e-9
lex = 5e-9

gridsize(128, 32, 1)
uniform(1, 1, 0)

#NOTE: gridsize should not be too small, makes time stepping crash
#due to exchange in convolution??

for i in range(10,20):
	d=i*lex
	partsize(5*d, d, 0.1*d)
	run(0.1e-9)
	save("m", "omf")
	save("table", "ascii")


