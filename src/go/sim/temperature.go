//  This file is part of MuMax, a high-performance micromagnetic simulator
//  Copyright 2010  Arne Vansteenkiste, Ben Van de Wiele
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package sim

// This file implements thermal fluctuations.

import (
)


// Accoding to Brown:
// H = η sqrt( 2 α kB T / γ μ0 Ms V dt )
func (s *Sim) addThermalNoise(h *DevTensor){
	s.assureTempInitiated()
	
}


func (s *Sim) assureTempInitiated(){
	if (s.tempNoise == nil){
		s.tempNoise = NewTensor(s.Backend, s.size3D)
	}	
}
