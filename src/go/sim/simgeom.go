package sim

// This file implements the methods for
// controlling the simulation geometry.

// Set the mesh size (number of cells in each direction)
// Note: for performance reasons the last size should be big.
// TODO: if the above is not the case, transparently transpose.
func (s *Sim) Size(x, y, z int) {
	s.size[X] = x
	s.size[Y] = y
	s.size[Z] = z
	s.invalidate()
}

// Defines the cell size in meters
func (s *Sim) CellSize(x, y, z float) {
	s.cellsize[X] = x
	s.cellsize[Y] = y
	s.cellsize[Z] = z
	s.invalidate()
}

// TODO: Defining the overall size and the (perhaps maximum) cell size,
// and letting the program choose the number of cells would be handy.
