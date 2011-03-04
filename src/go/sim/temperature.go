//  This file is part of MuMax, a high-performance micromagnetic simulator
//  Copyright 2010  Arne Vansteenkiste, Ben Van de Wiele
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package sim

// This file implements thermal fluctuations.

import (
	. "mumax/common"
	"os"
	"fmt"
)


// Accoding to Brown:
// H = η sqrt( 2 α kB T / γ μ0 Ms V dt )
func (s *Sim) addThermalField(h *DevTensor){
	s.assureTempInitiated()
	dt := s.dt * s.UnitTime() // TODO: check that this dt is up to date when adapted by solver...
	V := s.input.cellSize[X] *s.input.cellSize[Y] *  s.input.cellSize[Z]
	stddev := Sqrt32( (2 * s.alpha * kBSI * s.input.temp) / (s.gamma0 * s.mu0 * s.mSat * V * dt) )	/  s.UnitField()
	fmt.Fprintln(os.Stderr, "stddev: ", stddev)
	for i:=0; i<3; i++{
		s.GaussianNoise(s.tempNoise, stddev)
		s.Add( h.comp[i], s.tempNoise)
	}
}


func (s *Sim) assureTempInitiated(){
	if (s.tempNoise == nil){
		s.tempNoise = NewTensor(s.Backend, s.size3D)
	}
}


func (s *Sim) Temperature(T float32){
	s.input.temp = T
	s.Println("Temperature = ", T, " K")
}
