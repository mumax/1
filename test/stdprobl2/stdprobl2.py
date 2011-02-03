from mumax import *

# we chose lex fixed to 1

msat(800e3)
aexch(1.3e-11)
alpha(1)

#lex = 5.6858e-9
lex = 5e-9
d=10*lex

partsize(5*d, d, 0.1*d)
gridsize(128, 32, 1)
uniform(1, 1, 0)

run(1e-9)
save("m", "omf")

