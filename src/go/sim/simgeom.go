//  Copyright 2010  Arne Vansteenkiste
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package sim

import (
	"tensor"
	"os"
	"fmt"
)

// This file implements the methods for
// controlling the simulation geometry.

// Set the mesh size (number of cells in each direction)
// Note: for performance reasons the last size should be big.
// TODO: if the above is not the case, transparently transpose.
func (s *Sim) Size(x, y, z int) {
	if x <= 0 || y <= 0 || z <= 0 {
		s.Errorln("Size should be > 0")
		os.Exit(-6)
	}
	if x > y || x > z {
		s.Warn("For optimal efficiency, the number of cells in the first dimension (X) should be the smallest.\n E.g.: size 1 32 32 is much faster than size 32 32 1")
	}
	s.input.size[X] = x
	s.input.size[Y] = y
	s.input.size[Z] = z
	s.invalidate()
}

// Defines the cell size in meters
func (s *Sim) CellSize(x, y, z float32) {
	if x <= 0. || y <= 0. || z <= 0. {
		s.Errorln("Cell size should be > 0")
		os.Exit(-6)
	}
	s.input.cellSize[X] = x
	s.input.cellSize[Y] = y
	s.input.cellSize[Z] = z
	s.metadata["cellsize0"] = fmt.Sprint(x)
	s.metadata["cellsize1"] = fmt.Sprint(y)
	s.metadata["cellsize2"] = fmt.Sprint(z)
	s.invalidate()
}

func (sim *Sim) initNormMap() {
	if sim.normMap == nil {
		sim.normMap = NewTensor(sim.backend, Size3D(sim.mLocal.Size()))
	}
}

func (sim *Sim) LoadMSat(file string) {
	sim.initNormMap()
	sim.Println("Loading space-dependent saturation magnetization (norm)", file)
	in, err := os.Open(file, os.O_RDONLY, 0666)
	defer in.Close()
	if err != nil {
		panic(err)
	}
	norm := tensor.ToT3(tensor.Read(in))
	if !tensor.EqualSize(norm.Size(), sim.normMap.Size()){
    norm = resample3(norm, sim.normMap.Size())
  }
	TensorCopyTo(norm, sim.normMap)
	//TODO this should not invalidate the entire sim
	sim.invalidate()
}

// Sets up the normMap for a (possibly ellipsoidal) cylinder geometry along Z.
// Does not take into account the aspect ratio of the cells.
func (sim *Sim) Cylinder() {
	sim.initMLocal()
	sim.Println("Geometry: cylinder")

	sim.initNormMap()
	norm := tensor.NewT3(sim.normMap.Size())

	sizex := sim.mLocal.Size()[1]
	sizey := sim.mLocal.Size()[2]
	sizez := sim.mLocal.Size()[3]
	rx := float64(sizey / 2)
	ry := float64(sizez / 2)

	for i := 0; i < sizex; i++ {
		for j := 0; j < sizey; j++ {
			x := float64(j-sizey/2) + 0.5 // add 0.5 to be at the center of the cell, not a vertex (gives nicer round shape)
			for k := 0; k < sizez; k++ {
				y := float64(k-sizez/2) + 0.5
				if sqr(x/rx)+sqr(y/ry) <= 1 {
					norm.Array()[i][j][k] = 1.
				} else {
					norm.Array()[i][j][k] = 0.
				}

			}
		}
	}
	TensorCopyTo(norm, sim.normMap)
}

func sqr(x float64) float64 {
	return x * x
}

// TODO: Defining the overall size and the (perhaps maximum) cell size,
// and letting the program choose the number of cells would be handy.
