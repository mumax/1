//  Copyright 2010  Arne Vansteenkiste
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any
//  copyright notices and prominently state that you modified it, giving a relevant date.

package sim

import (
	. "mumax/common"
	"math"
)

/*
  Input methods for anisotropy
*/

func (s *Sim) K1(k1 float32) {
	s.input.anisKSI[0] = k1
	s.invalidate()
}

func (s *Sim) K2(k2 float32) {
	s.input.anisKSI[1] = k2
	s.invalidate()
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
	s.invalidate()
}

// Sets a cubic anisotropy
// K1 still needs to be set separately
// ux,uy,uz is the anisotropy direction,
// it does not need to be normalized
func (s *Sim) AnisCubic(u1z, u1y, u1x float32, u2z, u2y, u2x float32) {
	s.input.anisType = ANIS_CUBIC

	norm1 := sqrt32(u1x*u1x + u1y*u1y + u1z*u1z)
	norm2 := sqrt32(u2x*u2x + u2y*u2y + u2z*u2z)
	if norm1 == 0. || norm2 == 0. {
		panic(InputErr("Anisotropy axis should not be 0"))
	}
	u1x /= norm1
	u1y /= norm1
	u1z /= norm1

	u2x /= norm1
	u2y /= norm1
	u2z /= norm1

	u3x := u1y*u2z - u1z*u2y
	u3y := u1x*u2z - u1z*u2x
	u3z := u1x*u2y - u1y*u2x

	s.Println("Cubic anisotropy axes: (", u1z, u1y, u1x, "), (", u2z, u2y, u2x, "), (", u3z, u3y, u3x, ")")

	s.input.anisAxes = []float32{u1x, u1y, u1z, u2x, u2y, u2z, u3x, u3y, u3z}
	s.invalidate()
}

const (
	ANIS_NONE     = 0
	ANIS_UNIAXIAL = 1
	ANIS_CUBIC    = 2
)

func sqrt32(x float32) float32 {
	return float32(math.Sqrt(float64(x)))
}
