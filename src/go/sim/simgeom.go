//  This file is part of MuMax, a high-performance micromagnetic simulator
//  Copyright 2010  Arne Vansteenkiste
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package sim

import (
	. "mumax/common"
	"os"
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


// Defines the cell size in meters
func (s *Sim) CellSize(z, y, x float32) {
	if x <= 0. || y <= 0. || z <= 0. {
		s.Errorln("Cell size should be > 0")
		os.Exit(-6)
	}
	s.input.cellSize[X] = x
	s.input.cellSize[Y] = y
	s.input.cellSize[Z] = z
	s.input.cellSizeSet = true
	s.updateSizes()
	s.invalidate()
}

// Defines the maximum cell size in meters
func (s *Sim) MaxCellSize(z, y, x float32) {
	if x <= 0. || y <= 0. || z <= 0. {
		s.Errorln("Max cell size should be > 0")
		os.Exit(-6)
	}
	s.input.maxCellSize[X] = x
	s.input.maxCellSize[Y] = y
	s.input.maxCellSize[Z] = z
	s.input.maxCellSizeSet = true
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
	s.input.partSizeSet = true
	s.updateSizes()
	s.invalidate()
}


func (s *Sim) Periodic(z, y, x int) {
	s.input.periodic[X] = x
	s.input.periodic[Y] = y
	s.input.periodic[Z] = z
	s.invalidate()
}

// Input file may set only 2 values in the set {size, cellSize, partSize}. The last one is calculated here. It is an error to set all 3 of them.
func (s *Sim) updateSizes() {
	in := &s.input
	defer s.invalidate()

	// Check if two of the four size options have been set
	numset := 0
	for _, set := range []bool{in.sizeSet, in.cellSizeSet, in.partSizeSet, in.maxCellSizeSet} {
		if set {
			numset++
		}
	}
	if numset > 2 {
		panic(InputErr("Exactly two of [size, cellsize, partsize, maxcellsize] must be specified"))
	}

	if in.sizeSet && in.cellSizeSet {
		for i := range in.partSize {
			in.partSize[i] = float32(in.size[i]) * in.cellSize[i]
		}
		s.Println("Calculated part size:", in.partSize, " m")
		return
	}

	if in.sizeSet && in.partSizeSet {
		for i := range in.partSize {
			in.cellSize[i] = in.partSize[i] / float32(in.size[i])
		}
		s.Println("Calculated cell size:", in.cellSize, " m")
		return
	}

	if in.cellSizeSet && in.partSizeSet {
		for i := range in.partSize {
			in.size[i] = int((in.partSize[i] / in.cellSize[i]) + 0.5) // We round as good as possible, it is up to the user for partsize to be divisible by cellsize
		}
		s.Println("Calculated number of cells:", in.size)
		return
	}

	// Find a gridsize that is suited for CUFFT and
	// so that the cell size does not exceed maxCellSize by more than a few %. 
	if in.maxCellSizeSet && in.partSizeSet {
		for i := range in.partSize {
			n := int(in.partSize[i] / (in.maxCellSize[i] * (1 + MAX_OVERSIZE)))
			for !IsGoodGridSize(i, n) { // direction-dependent
				n++
			}
			in.size[i] = n
			in.cellSize[i] = in.partSize[i] / float32(in.size[i])
		}
		s.Println("Calculated number of cells:", in.size)
		return
	}

	//panic(InputErr("A valid combination of [size, cellsize, partsize, maxcellsize] must be specified"))

	// s.invalidate() deferred
}

// When calculating a suited grid size from a maximum cell size,
// make cells at most this fraction bigger than the specified maximum.
// (It's better to make them a few percent bigger than a factor 2 too small, e.g.)
const MAX_OVERSIZE = 0.05

// Returns the smallest power of two >= n
func findPow2(n float32) int {
	if n < 1.0 {
		n = 1.0
	}
	return int(math.Pow(2, math.Ceil(math.Log2(float64(n)))))

}

// Sets the accuracy of edge corrections.
// 0 means no correction.
func (s *Sim) EdgeCorrection(accuracy int) {
	s.input.edgeCorr = accuracy
	s.invalidate()
}


//func (sim *Sim) LoadMSat(file string) {
//	sim.allocNormMap()
//	sim.Println("Loading space-dependent saturation magnetization (norm)", file)
//	in, err := os.Open(file, os.O_RDONLY, 0666)
//	defer in.Close()
//	if err != nil {
//		panic(err)
//	}
//	norm := tensor.ToT3(tensor.Read(in))
//	if !tensor.EqualSize(norm.Size(), sim.normMap.Size()) {
//		norm = resample3(norm, sim.normMap.Size())
//	}
//	TensorCopyTo(norm, sim.normMap)
//	sim.updateAvgNorm()
//	//TODO this should not invalidate the entire sim
//	sim.invalidate()
//}

var INF32 float32 = float32(math.Inf(1))

// Sets up the normMap for a (possibly ellipsoidal) cylinder geometry along Z.
// Does not take into account the aspect ratio of the cells.
//func (sim *Sim) Cylinder() {
//	sim.initSize()
//	sim.input.geom = &Ellipsoid{INF32, sim.input.partSize[Y] / 2., sim.input.partSize[Z] / 2.}
//	sim.invalidate()
//}


func (sim *Sim) DotArray(r, sep float32, n int) {
	pitch := 2*r + sep
	sim.input.geom = &Array{&Ellipsoid{INF32, r, r}, n, n, pitch, pitch}
}

func (sim *Sim) SquareHoleArray(r, sep float32, n int) {
	pitch := 2*r + sep
	sim.input.geom = &Inverse{&Array{&Cuboid{INF32, r, r}, n, n, pitch, pitch}}
}

func (sim *Sim) Ellipsoid(rz, ry, rx float32) {
	sim.input.geom = &Ellipsoid{rx, ry, rz}
	sim.invalidate()
}


//DEBUG
// func (s *Sim) TestGeom(w, h float32) {
// 	s.initSize()
// 	s.geom = &Wave{w, h}
// 	s.invalidate()
// }

func sqr(x float64) float64 {
	return x * x
}

var inf float32 = float32(math.Inf(1))


func (sim *Sim) initAlphaMask() {
	if sim.alphaMask == nil {
		sim.alphaMask = NewTensor(sim.Backend, sim.size[:])
		for i := 0; i < sim.size[0]; i++ {
			for j := 0; j < sim.size[1]; j++ {
				for k := 0; k < sim.size[2]; k++ {
					sim.alphaMask.Set(i, j, k, 1)
				}
			}
		}
	}
}

// Set the damping constant in a cell [x, y, z]
func (sim *Sim) SetAlpha(z, y, x int, alpha float32) {
	sim.initGeom()
	sim.initAlphaMask()
	sim.alphaMask.Set(x, y, z, alpha)
}

// Set the damping constant in an area between [x1, y1, z1] and [x2, y2, z2] (inclusive)
func (sim *Sim) SetAlphaRange(x1, y1, z1, x2, y2, z2 int, alpha float32) {
	sim.initGeom()
	sim.initAlphaMask()

	for i := z1; i <= z2; i++ {
		for j := y1; j <= y2; j++ {
			for k := x1; k <= x2; k++ {
				sim.alphaMask.Set(i, j, k, alpha)
			}
		}
	}
}

func (sim *Sim) initMsatMask() {
	if sim.normMap == nil {
		sim.normMap = NewTensor(sim.Backend, sim.size[:])
		for i := 0; i < sim.size[0]; i++ {
			for j := 0; j < sim.size[1]; j++ {
				for k := 0; k < sim.size[2]; k++ {
					sim.normMap.Set(i, j, k, 1)
				}
			}
		}
	}
}

// Set the normalized (!) saturation magnetization in a cell [x, y, z]
func (sim *Sim) SetMsat(z, y, x int, msat float32) {
	sim.initGeom()
	sim.initMsatMask()
	sim.normMap.Set(x, y, z, msat)
}


// Set the normalized (!) saturation magnetization in an rectangular area between [x1, y1, z1] and [x2, y2, z2] (inclusive)
func (sim *Sim) SetMsatRange(z1, y1, x1, z2, y2, x2 int, msat float32) {
	sim.initGeom()
	sim.initMsatMask()

	for i := x1; i <= x2; i++ {
		for j := y1; j <= y2; j++ {
			for k := z1; k <= z2; k++ {
				sim.normMap.Set(i, j, k, msat)
			}
		}
	}
}

// Set the normalized (!) saturation magnetization in an ellips-shaped region with center [cx, cy] and semi-axes [sx, sy]
func (s *Sim) SetMsatEllips(cx, cy, sx, sy, msat float32) {

	s.initGeom()
	s.initMsatMask()
	rx := sx/2.0
	ry := sy/2.0

	c := s.input.cellSize
	start_x := int(math.Floor(float64((cx - rx) / c[2])))
	start_y := int(math.Floor(float64((cy - ry) / c[1])))
	stop_x := int(math.Ceil(float64((cx + rx) / c[2])))
	stop_y := int(math.Ceil(float64((cy + ry) / c[1])))

	for i := 0; i < s.size[0]; i++ {
		for j := start_y; j <= stop_y; j++ {
			y := float32(j)*c[1] - cy
			for k := start_x; k <= stop_x; k++ {
				x := float32(k)*c[2] - cx
				if (x*x/rx/rx + y*y/ry/ry) < 1 {
					s.normMap.Set(i, j, k, msat)
				}
			}
		}
	}

	/*for i := 0; i<s.size[0]; i++ {
	  for j := start_y; j<=stop_y; j++  {
	    y := float32(j)*c[1] - cy
	    for k := start_x; j<=stop_x; k++ {
	      x := float32(k)*c[2] - cx
	      if (x*x/rx/rx + y*y/ry/ry)<1 {
	        s.normMap.Set(i,j,k, 0.0)
	      }
	    }
	  }
	}*/

}

// Make an dot array with Nx times Ny rectangular holes. The unit cells (separation + dot) have dimensions unit_size_x times unit_size_y in nanometer. 
func (sim *Sim) DotArrayRectangle(unit_size_x, unit_size_y, separation_x, separation_y float32, Nx, Ny int) {

	sim.initGeom()
	sim.initMsatMask()

	c := sim.input.cellSize
	Bx := unit_size_x / c[2]
	Sx := separation_x / c[2]
	By := unit_size_y / c[1]
	Sy := separation_y / c[1]

	for cnt1 := 0; cnt1 < Ny; cnt1++ {
		for cnt2 := 0; cnt2 < Nx; cnt2++ {
			for i := 0; i < sim.size[0]; i++ {
				for j := int(float32(cnt1) * By); j < int(float32(cnt1+1)*By); j++ {
					for k := int(float32(cnt2) * Bx); k < int(float32(cnt2+1)*Bx); k++ {
						if !(j >= int(float32(cnt1)*By+Sy/2) && j < int(float32(cnt1+1)*By-Sy/2) && k >= int(float32(cnt2)*Bx+Sx/2) && k < int(float32(cnt2+1)*Bx-Sx/2)) {
							sim.normMap.Set(i, j, k, 0.0)
						}
					}
				}
			}
		}
	}
}

// Make an dot array with Nx times Ny ellips-shaped holes. The unit cells (separation + dot) have dimensions unit_size_x times unit_size_y in nanometer. 
func (s *Sim) DotArrayEllips(unit_size_x, unit_size_y, separation_x, separation_y float32, Nx, Ny int) {

	s.initGeom()
	s.initMsatMask()

	c := s.input.cellSize
	Bx := unit_size_x / c[2]
	Sx := separation_x / c[2]
	rx := (Bx - Sx) / 2.
	By := unit_size_y / c[1]
	Sy := separation_y / c[1]
	ry := (By - Sy) / 2.

	for cnt1 := 0; cnt1 < Ny; cnt1++ {
		cy := float32(cnt1)*By + By/2 - 0.5
		for cnt2 := 0; cnt2 < Nx; cnt2++ {
			cx := float32(cnt2)*Bx + Bx/2 - 0.5
			for i := 0; i < s.size[0]; i++ {
				for j := int(float32(cnt1) * By); j < int(float32(cnt1+1)*By); j++ {
					y := float32(j) - cy
					for k := int(float32(cnt2) * Bx); k < int(float32(cnt2+1)*Bx); k++ {
						x := float32(k) - cx
						if (x*x/rx/rx + y*y/ry/ry) > 1 {
							s.normMap.Set(i, j, k, 0.0)
						}
					}
				}
			}
		}
	}
}

// Make an anti-dot array with Nx times Ny rectangular holes. The unit cells (separation + hole) have dimensions unit_size_x times unit_size_y in nanometer. 
func (sim *Sim) AntiDotArrayRectangle(unit_size_x, unit_size_y, separation_x, separation_y float32, Nx, Ny int) {

	sim.initGeom()
	sim.initMsatMask()

	c := sim.input.cellSize
	Bx := unit_size_x / c[2]
	Sx := separation_x / c[2]
	By := unit_size_y / c[1]
	Sy := separation_y / c[1]

	for cnt1 := 0; cnt1 < Ny; cnt1++ {
		for cnt2 := 0; cnt2 < Nx; cnt2++ {
			for i := 0; i < sim.size[0]; i++ {
				for j := int(float32(cnt1)*By + Sy/2); j < int(float32(cnt1+1)*By-Sy/2); j++ {
					for k := int(float32(cnt2)*Bx + Sx/2); k < int(float32(cnt2+1)*Bx-Sx/2); k++ {
						sim.normMap.Set(i, j, k, 0.0)
					}
				}
			}
		}
	}
}


// Make an anti-dot array with Nx times Ny ellips-shaped holes. The basic cells (separation + hole) have dimensions unit_size_x times unit_size_y in nanometer. 
func (s *Sim) AntiDotArrayEllips(unit_size_x, unit_size_y, separation_x, separation_y float32, Nx, Ny int) {

	s.initGeom()
	s.initMsatMask()

	c := s.input.cellSize
	Bx := unit_size_x / c[2]
	Sx := separation_x / c[2]
	rx := (Bx - Sx) / 2.
	By := unit_size_y / c[1]
	Sy := separation_y / c[1]
	ry := (By - Sy) / 2.

	for cnt1 := 0; cnt1 < Ny; cnt1++ {
		cy := float32(cnt1)*By + By/2 - 0.5
		for cnt2 := 0; cnt2 < Nx; cnt2++ {
			cx := float32(cnt2)*Bx + Bx/2 - 0.5
			for i := 0; i < s.size[0]; i++ {
				for j := int(float32(cnt1)*By + Sy/2); j < int(float32(cnt1+1)*By-Sy/2); j++ {
					y := float32(j) - cy
					for k := int(float32(cnt2)*Bx + Sx/2); k < int(float32(cnt2+1)*Bx-Sx/2); k++ {
						x := float32(k) - cx
						if (x*x/rx/rx + y*y/ry/ry) < 1 {
							s.normMap.Set(i, j, k, 0.0)
						}
					}
				}
			}
		}
	}
}
