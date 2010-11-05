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
	"math"
)

// This file implements the methods for
// controlling the simulation geometry.

//                                                                      IMPORTANT: this is one of the places where X,Y,Z get swapped
//                                                                      what is (X,Y,Z) internally becomes (Z,Y,X) for the user!


// Set the mesh size (number of cells in each direction)
// Note: for performance reasons the last size should be big.
func (s *Sim) GridSize(z, y, x int) {
	if x <= 0 || y <= 0 || z <= 0 {
		s.Errorln("Size should be > 0")
		os.Exit(-6)
	}
	if x > y || x > z {
		s.Warn("For optimal efficiency, the number of cells in the last dimension (Z) should be the smallest.\n E.g.: gridsize 32 32 1 is much faster than gridsize 1 32 32")
	}
	s.input.size[X] = x
	s.input.size[Y] = y
	s.input.size[Z] = z
	s.input.sizeSet = true
	s.updateSizes()
	s.invalidate()
}

// TODO: We need one function that sets all metadata centrally

// Defines the cell size in meters
func (s *Sim) CellSize(z, y, x float32) {
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
	s.input.cellSizeSet = true
	s.updateSizes()
	s.invalidate()
}

func (s *Sim) PartSize(z, y, x float32) {
	if x <= 0. || y <= 0. || z <= 0. {
		s.Errorln("Part size should be > 0")
		os.Exit(-6)
	}
	s.input.partSize[X] = x
	s.input.partSize[Y] = y
	s.input.partSize[Z] = z
	s.metadata["partsize0"] = fmt.Sprint(x)
	s.metadata["partsize1"] = fmt.Sprint(y)
	s.metadata["partsize2"] = fmt.Sprint(z)
	s.input.partSizeSet = true
	s.updateSizes()
	s.invalidate()
}

// Input file may set only 2 values in the set {size, cellSize, partSize}. The last one is calculated here. It is an error to set all 3 of them.
func (s *Sim) updateSizes() {
	in := &s.input

	if in.sizeSet && in.cellSizeSet && in.partSizeSet {
		panic(InputErr("size, cellsize and partsize may not all be specified together. Specify any two of them and the third one will be calculated automatically."))
	}

	if in.sizeSet && in.cellSizeSet {
		for i := range in.partSize {
			in.partSize[i] = float32(in.size[i]) * in.cellSize[i]
		}
		s.Println("Calculated part size:", in.partSize, " m")
	}

	if in.sizeSet && in.partSizeSet {
		for i := range in.partSize {
			in.cellSize[i] = in.partSize[i] / float32(in.size[i])
		}
		s.Println("Calculated cell size:", in.cellSize, " m")
	}

	if in.cellSizeSet && in.partSizeSet {
		for i := range in.partSize {
			in.size[i] = int((in.partSize[i] / in.cellSize[i]) + 0.5) // We round as good as possible, it is up to the user for partsize to be divisible by cellsize
		}
		s.Println("Calculated number of cells:", in.size)
	}

	s.invalidate()
}


// Sets the accuracy of edge corrections.
// 0 means no correction.
func (s *Sim) EdgeCorrection(accuracy int) {
	s.edgecorr = accuracy
	s.invalidate()
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
	if !tensor.EqualSize(norm.Size(), sim.normMap.Size()) {
		norm = resample3(norm, sim.normMap.Size())
	}
	TensorCopyTo(norm, sim.normMap)
	//TODO this should not invalidate the entire sim
	sim.invalidate()
}

// Sets up the normMap for a (possibly ellipsoidal) cylinder geometry along Z.
// Does not take into account the aspect ratio of the cells.
func (sim *Sim) Cylinder() {

	sim.initSize()
	sim.geom = &Ellipsoid{float32(math.Inf(1)), sim.input.partSize[Y] / 2., sim.input.partSize[Z] / 2.}
	sim.invalidate()

	// 	sim.initMLocal()
	// 	sim.Println("Geometry: cylinder")
	// 
	// 	sim.initNormMap()
	// 	norm := tensor.NewT3(sim.normMap.Size())
	// 
	// 	sizex := sim.mLocal.Size()[1]
	// 	sizey := sim.mLocal.Size()[2]
	// 	sizez := sim.mLocal.Size()[3]
	// 	rx := float64(sizey / 2)
	// 	ry := float64(sizez / 2)
	// 
	// 	for i := 0; i < sizex; i++ {
	// 		for j := 0; j < sizey; j++ {
	// 			x := float64(j-sizey/2) + 0.5 // add 0.5 to be at the center of the cell, not at a corner vertex (gives nicer round shape)
	// 			for k := 0; k < sizez; k++ {
	// 				y := float64(k-sizez/2) + 0.5
	// 				if sqr(x/rx)+sqr(y/ry) <= 1 {
	// 					norm.Array()[i][j][k] = 1.
	// 				} else {
	// 					norm.Array()[i][j][k] = 0.
	// 				}
	// 
	// 			}
	// 		}
	// 	}
	// 	TensorCopyTo(norm, sim.normMap)
}

func sqr(x float64) float64 {
	return x * x
}
