//  Copyright 2010  Arne Vansteenkiste
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

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
