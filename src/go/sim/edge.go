//  Copyright 2010  Arne Vansteenkiste
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any
//  copyright notices and prominently state that you modified it, giving a relevant date.

package sim

import (
	"tensor"
)

func (s *Sim) initGeom() {

	s.initMLocal()

	if s.geom == nil {
		s.Println("Square geometry")
		return
	}

	s.initNormMap()

	norm := tensor.NewT3(s.normMap.Size())

	sizex := s.mLocal.Size()[1]
	sizey := s.mLocal.Size()[2]
	sizez := s.mLocal.Size()[3]
	
  centerx := 0.5 * (s.input.partSize[X])
  centery := 0.5 * (s.input.partSize[Y])
  centerz := 0.5 * (s.input.partSize[Z])


	for i := 0; i < sizex; i++ {
    x := (float32(i)+.5) * (s.input.partSize[X] / float32(sizex)) - centerx
		for j := 0; j < sizey; j++ {
		y := (float32(j)+.5) * (s.input.partSize[Y] / float32(sizey)) - centery
			for k := 0; k < sizez; k++ {
			z := (float32(k)+.5) * (s.input.partSize[Z] / float32(sizez)) -centerz
				if s.geom.Inside(x, y, z) {
					norm.Array()[i][j][k] = 1.
				} else {
					norm.Array()[i][j][k] = 0.
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


func (sim *Sim) initNormMap() {
	sim.initMLocal()
	if sim.normMap == nil {
		sim.normMap = NewTensor(sim.Backend, Size3D(sim.mLocal.Size()))
	}
}
