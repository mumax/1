//  Copyright 2010  Arne Vansteenkiste
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any
//  copyright notices and prominently state that you modified it, giving a relevant date.

package sim

import (
	"tensor"
)

// TODO: what if there is already an msat map that represents variations
// in the material, not geometry? It should not just be overwritten...
func (s *Sim) initGeom() {

	s.initMLocal()

	if s.geom == nil {
		s.Println("Square geometry")
		return
	}

	// (1) Initialize and calculate the norm map
	// (norm of msat)
	s.allocNormMap()
	norm := tensor.NewT3(s.normMap.Size()) // local copy

	// Even without edge corrections it is good to have soft (antialiased) edges
	// Normally we use the same subsampling accuracy as required by the edge corrections,
	// but when edgecorrection==0, we use a default smoothness.
	softness := s.edgecorr
	if softness == 0 {
		softness = 3
		s.Println("Using default edge smoothness: 2^", softness)
	}

	refine := pow(2, edgecorr) // use refine x refine x refine subcells per cell. refine = 2^edgecorr
	s.Println("Edge refinement: ", refine)
	refine3 := float32(pow(refine, 3))

	// Loop over the refined grid, it is larger than the actual simulation grid
	sizex := s.mLocal.Size()[1] * refine
	sizey := s.mLocal.Size()[2] * refine
	sizez := s.mLocal.Size()[3] * refine

	// Count how many of the sub-cells lie inside the geometry
	// The normMap for the cell will be between 0 and 1 depending
	// on the portion of the cell that lies inside the geometry
	// TODO: can be optimized for 2D
	for i := 0; i < sizex; i++ {
		x := (float32(i)+.5)*(s.input.partSize[X]/float32(sizex)) - 0.5*(s.input.partSize[X]) // fine coordinate inside the magnet, SI units
		for j := 0; j < sizey; j++ {
			y := (float32(j)+.5)*(s.input.partSize[Y]/float32(sizey)) - 0.5*(s.input.partSize[Y])
			for k := 0; k < sizez; k++ {
				z := (float32(k)+.5)*(s.input.partSize[Z]/float32(sizez)) - 0.5*(s.input.partSize[Z])

				if s.geom.Inside(x, y, z) {
					norm.Array()[i/refine][j/refine][k/refine] += 1. / refine3
				}

			}
		}
	}
	TensorCopyTo(norm, s.normMap)

	if s.edgecorr == 0 {
		return
	}

	s.Println("Initializing edge corrections")

}


func (sim *Sim) allocNormMap() {
	sim.initMLocal()
	if sim.normMap == nil {
		sim.normMap = NewTensor(sim.Backend, Size3D(sim.mLocal.Size()))
	}
}


func pow(base, exp int) int {
	result := 1
	for i := 0; i < exp; i++ {
		result *= base
	}
	return result
}
