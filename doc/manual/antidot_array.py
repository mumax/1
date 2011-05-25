# This example shows how to define an antidot array

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

uniform (1, 0, 0)

# for an antidot array with rectangular holes
antiDotArrayRectangle(unit_size_x, unit_size_y, separation_x, separation_y, Ndots_x, Ndots_y)

# for an antidot array with ellips-shaped holes
#antiDotArrayEllips(unit_size_x, unit_size_y, separation_x, separation_y, Ndots_x, Ndots_y)

step()

save("m", "png")
save("m", "ascii")



