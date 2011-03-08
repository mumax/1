//  This file is part of MuMax, a high-performance micromagnetic simulator
//  Copyright 2010  Arne Vansteenkiste
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package sim

// This file implements functions for calculating the effective field.

import (
	. "mumax/common"
)

// Calulates the sum of the demag and exchange fields.
func (s *Sim) calcHDemagExch(m, h *DevTensor) {
	if s.input.wantDemag {
		s.Convolve(m, h)
	} else {
		ZeroTensor(h)
	}
	if !s.input.wantDemag || !s.exchInConv || IsInf(s.cellSize[X]) {
		s.AddExch(m, h)
	}
}

// Adds the "local" fields to H (zeeman, anisotropy)
func (s *Sim) addLocalFields(m, h *DevTensor) {
	if s.appliedField != nil {
		s.hextSI = s.appliedField.GetAppliedField(s.time * float64(s.UnitTime()))
	} else {
		s.hextSI = [3]float32{0., 0., 0.}
	}

	B := s.UnitField()
	s.hextInt[0] = s.hextSI[0] / B
	s.hextInt[1] = s.hextSI[1] / B
	s.hextInt[2] = s.hextSI[2] / B

	// TODO: only if needed
	s.AddLocalFields(m, h, s.hextInt, s.input.anisType, s.anisKInt, s.input.anisAxes)
}

// Calculates the effective field of m and stores it in h
func (s *Sim) calcHeff(m, h *DevTensor) {
	// (1) Self-magnetostatic field and exchange
	// The convolution may include the exchange field
	s.calcHDemagExch(m, h)

	// (2) Add the externally applied field
	s.addLocalFields(m, h)

	if s.input.temp != 0 {
		s.addThermalField(h)
	}

	// (3) Add the edge-correction field
	if s.input.edgeCorr != 0 {
		s.addEdgeField(m, h)
	}
}
