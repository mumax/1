from mumax import *

# we chose lex fixed to 1

msat(800e3)
aexch(1.3e-11)
alpha(1)

lex = 5.6858e-9
d=20*lex
partsize(5*d, d, 0.1*d)
maxcellsize(lex, lex, inf)
uniform(1, 1, 0)

for i in range(0, 100):
	staticfield(-i*1e-3, 0, 0)
	run(100e-12)
	relax()
	save("m", "png")
	save("table", "ascii")



