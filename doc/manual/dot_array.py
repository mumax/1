# This example shows how to define an dotarray.

from mumax import *

# material
msat(800e3)
aexch(13.0e-12)
alpha(1)

thickness = 10e-9
cell_size = 3.125e-9

separation_x = 4*cell_size
separation_y = 8*cell_size

hole_x = 12*cell_size
hole_y = 8*cell_size

unit_size_x = separation_x + hole_x
unit_size_y = separation_y + hole_y

Ndots_x = 8
Ndots_y = 4


cellSize( cell_size, cell_size, thickness )
gridSize( 128, 64, 1 )

uniform (0, 1, 0)

# for an antidot array with rectangular holes
# dotArrayRectangle(unit_size_x, unit_size_y, separation_x, separation_y, Ndots_x, Ndots_y)

# for an antidot array with ellips-shaped holes
# dotArrayEllips(unit_size_x, unit_size_y, separation_x, separation_y, Ndots_x, Ndots_y)

setMsatEllips(10*cell_size, 12*cell_size, 6*cell_size, 4*cell_size, 0.0)

# normalization of the magnetization is only performed at the beginning of a timestep
step()

save("m", "png")
save("m", "ascii")



