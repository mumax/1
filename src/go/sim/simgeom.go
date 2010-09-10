package sim

// This file implements the methods for
// controlling the simulation geometry.

// Set the mesh size (number of cells in each direction)
// Note: for performance reasons the last size should be big.
// TODO: if the above is not the case, transparently transpose.
func (s *Sim) Size(x, y, z int) {
	s.input.size[X] = x
	s.input.size[Y] = y
	s.input.size[Z] = z
	s.invalidate()
}

// Defines the cell size in meters
func (s *Sim) CellSize(x, y, z float) {
	s.input.cellSize[X] = x
	s.input.cellSize[Y] = y
	s.input.cellSize[Z] = z
	s.invalidate()
}

// TODO: Defining the overall size and the (perhaps maximum) cell size,
// and letting the program choose the number of cells would be handy.
