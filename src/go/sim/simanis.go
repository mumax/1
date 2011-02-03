//  Copyright 2010  Arne Vansteenkiste
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any
//  copyright notices and prominently state that you modified it, giving a relevant date.

package sim

import "math"


/*
  Input methods for anisotropy
*/

func (s *Sim) K1(k1 float32) {
	// hack: move to sim.input
	s.InitMaterial()
	s.input.anisK[0] = k1 / s.UnitEnergyDensity()
	//s.Println("Anisotropy k1 = ", s.input.anisK[0], " Msat")
	s.invalidate() // for debug
	// does not invalidate (?)
}

// Sets a uniaxial anisotropy
// K1 still needs to be set separately
// ux,uy,uz is the anisotropy direction,
// it does not need to be normalized
func (s *Sim) AnisUniaxial(uz, uy, ux float32) {
	s.input.anisType = ANIS_UNIAXIAL
	norm := sqrt32(ux*ux + uy*uy + uz*uz)
	if norm == 0. {
		panic(InputErr("Anisotropy axis should not be 0"))
	}
	ux /= norm
	uy /= norm
	uz /= norm

	s.input.anisAxes = []float32{ux, uy, uz}
	s.invalidate() // debug
	// does not invalidate (?)
}

const (
	ANIS_NONE     = 0
	ANIS_UNIAXIAL = 1
	ANIS_CUBIC    = 2
)

func sqrt32(x float32) float32 {
	return float32(math.Sqrt(float64(x)))
}
