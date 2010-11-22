//  Copyright 2010  Arne Vansteenkiste
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any
//  copyright notices and prominently state that you modified it, giving a relevant date.

package sim

import (
	"tensor"
	"fmt"
)


func (s *Sim) addEdgeField(m, h *DevTensor) {
	s.AddLinAnis(m, h, s.edgeKern)
}


// TODO: what if there is already an msat map that represents variations
// in the material, not geometry? It should not just be overwritten...
func (s *Sim) initGeom() {

	s.initMLocal() // inits sizes etc.

	if s.geom == nil {
		s.Println("Square geometry")
		return
	}

	s.initNormMap()

	edgeCorr := pow(2, s.edgeCorr)
	if edgeCorr == 1 {
		return
	}

	s.Println("Initializing edge corrections")

	s.allocEdgeKern()
	e := make([]*tensor.T3, 6)    // local copy of edge kernel
	E := make([][][][]float32, 6) // edge kernel as array
	for i := range s.edgeKern {
		e[i] = tensor.NewT3(Size3D(s.mLocal.Size()))
		E[i] = e[i].Array()
	}

	// the demag self-kernel for cuboid cells, as calculated by demag.go
	// for each non-cuboid cell, we will need to subtract it and replace it by the edge-corrected kernel
	selfK := [][]float32{
		selfKernel(X, s.cellSize[:], s.input.demag_accuracy),
		selfKernel(Y, s.cellSize[:], s.input.demag_accuracy),
		selfKernel(Z, s.cellSize[:], s.input.demag_accuracy)}

	subSize := []int{2 * edgeCorr, 2 * edgeCorr, 2 * edgeCorr} // 2*: zero-padding
	subCellSize := []float32{s.cellSize[X] / float32(edgeCorr), s.cellSize[Y] / float32(edgeCorr), s.cellSize[Z] / float32(edgeCorr)}
	NO_PBC := []int{0, 0, 0} // no periodic boundary conditins
	// kernel for the sub-cells within one cell.
	subK := FaceKernel6(subSize, subCellSize, s.input.demag_accuracy, NO_PBC)

	sizex := s.mLocal.Size()[1]
	sizey := s.mLocal.Size()[2]
	sizez := s.mLocal.Size()[3]

	count := 0

	insideCache := make([][][]bool, edgeCorr)
	for i := range insideCache {
		insideCache[i] = make([][]bool, edgeCorr)
		for j := range insideCache[i] {
			insideCache[i][j] = make([]bool, edgeCorr)
		}
	}

	for i := 0; i < sizex; i++ {
		for j := 0; j < sizey; j++ {
			for k := 0; k < sizez; k++ {

				for si := 0; si < edgeCorr; si++ {
					xs := (float32(i*edgeCorr+si)+.5)*(s.input.partSize[X]/float32(sizex*edgeCorr)) - 0.5*(s.input.partSize[X])
					for sj := 0; sj < edgeCorr; sj++ {
						ys := (float32(j*edgeCorr+sj)+.5)*(s.input.partSize[Y]/float32(sizey*edgeCorr)) - 0.5*(s.input.partSize[Y])
						for sk := 0; sk < edgeCorr; sk++ {
							zs := (float32(k*edgeCorr+sk)+.5)*(s.input.partSize[Z]/float32(sizez*edgeCorr)) - 0.5*(s.input.partSize[Z])

							insideCache[si][sj][sk] = s.geom.Inside(xs, ys, zs)
						}
					}
				}

				norm := s.normLocal.TArray[i][j][k]
				if norm > 0 && norm < 1 {
					count++
					fmt.Println("norm: ", norm)
					for S := 0; S < 3; S++ {
						for D := 0; D < 3; D++ {
							E[KernIdx[S][D]][i][j][k] = -selfK[S][D]

							for si := 0; si < edgeCorr; si++ {
								for sj := 0; sj < edgeCorr; sj++ {
									for sk := 0; sk < edgeCorr; sk++ {

										if insideCache[si][sj][sk] {

											for di := 0; di < edgeCorr; di++ {
												for dj := 0; dj < edgeCorr; dj++ {
													for dk := 0; dk < edgeCorr; dk++ {

														if insideCache[di][dj][dk] {
															E[KernIdx[S][D]][i][j][k] += subK[KernIdx[S][D]].TArray[wrap(di-si, 2*edgeCorr)][wrap(dj-sj, 2*edgeCorr)][wrap(dk-sk, 2*edgeCorr)] / norm
														}
													}
												}
											}
										}
									}
								}
							}
						}
					}
				}
			}
		}
	}
	s.Println(count, " cells with edge-corrections")
	for i := range e {
		TensorCopyTo(e[i], s.edgeKern[i])
	}

}


// Calculates normMap (the norm of MSat, per cell),
// based on the magnet geometry.
// The normMap for the cell will lie between 0 and 1 depending
// on the portion of the cell that lies inside the geometry
func (s *Sim) initNormMap() {

	s.allocNormMap()
	s.normLocal = tensor.NewT3(s.normMap.Size()) // local copy
	norm := s.normLocal

	// Even without edge corrections it is good to have soft (antialiased) edges
	// Normally we use the same subsampling accuracy as required by the edge corrections,
	// but when edgecorrection==0, we use a default smoothness.
	softness := s.edgeCorr
	if softness == 0 {
		softness = 0  // TODO Temporarily put softness to 0 as long as dynamics of unnormalized spins are not fixed
		s.Println("Using default edge smoothness: 2^", softness)
	}

	refine := pow(2, softness) // use refine x refine x refine subcells per cell. refine = 2^edgecorr
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
}


func (sim *Sim) allocNormMap() {
	sim.initMLocal()
	if sim.normMap == nil {
		sim.Println("Allocating Norm Map")
		sim.normMap = NewTensor(sim.Backend, Size3D(sim.mLocal.Size()))
	}
}

// TODO: we could save a lot of memory here for cubic/square cells:
// Kxx = Kyy = Kzz, ...
// just make them point to the same underlying storage
func (sim *Sim) allocEdgeKern() {
	sim.initMLocal()
	if sim.edgeKern == nil {
		sim.Println("Allocating Edge Kernel")
		sim.edgeKern = make([]*DevTensor, 6)
		for i := range sim.edgeKern {
			sim.edgeKern[i] = NewTensor(sim.Backend, Size3D(sim.mLocal.Size()))
		}
	}
}


func pow(base, exp int) int {
	result := 1
	for i := 0; i < exp; i++ {
		result *= base
	}
	return result
}
