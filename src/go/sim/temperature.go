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
	//"os"
	//"fmt"
)


// Adds the stored thermal noise to h
// it must be updated first with updateTempNoise()
func (s *Sim) addThermalField(h *DevTensor) {
	s.Add(h, s.tempNoise)
}


// Updates the stored thermal noise. To be called at the first stage of each time step
// Accoding to Brown:
// H = η sqrt( 2 α kB T / γ μ0 Ms V dt )
func (s *Sim) updateTempNoise(dt float32) {
	s.assureTempInitiated()
	dt *= s.UnitTime()
	V := s.input.cellSize[X] * s.input.cellSize[Y] * s.input.cellSize[Z]
	stddev := Sqrt32((2*s.alpha*kBSI*s.input.temp)/(s.gamma0*s.mu0*s.mSat*V*dt)) / s.mSat
	//fmt.Fprintln(os.Stderr, "stddev: ", stddev)
	s.GaussianNoise(s.tempNoise, stddev)
}

func (s *Sim) assureTempInitiated() {
	if s.tempNoise == nil {
		s.tempNoise = NewTensor(s.Backend, s.size4D[:])
	}
}


func (s *Sim) Temperature(T float32) {
	if !s.IsValid() { // not yet initiated, set rk12 behind the screens.
		s.input.solvertype = "rk12"
	} else {
		if s.input.solvertype != "rk12" {
			panic(InputErr("Finite temperature is only compatible with the RK12 solver type"))
		}
	}
	s.input.temp = T
	s.Println("Temperature = ", T, " K")
}
