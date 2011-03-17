//  This file is part of MuMax, a high-performance micromagnetic simulator
//  Copyright 2010  Arne Vansteenkiste
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package sim

// This file implements retrieving values from the simulation.

import (
	"fmt"
)

// INTERNAL: Send a value to stdout, to be recieved by a subprocess.
func Send(v ...interface{}) {
	fmt.Print("%")
	fmt.Println(v...)
}


// Gets an average magnetization component.
func (s *Sim) GetM(component int) {
	Send(s.getM(2 - component)) // translate to ZYX

}

func (s *Sim) getM(component int) float32 {
	s.init()
	return s.devsum.Reduce(s.mDev.comp[component]) / s.avgNorm
}


func (s *Sim) GetMaxM(component int) {
	Send(s.getMaxM(2 - component)) // translate to ZYX

}
func (s *Sim) getMaxM(component int) float32 {
	s.init()
	return s.devmax.Reduce(s.mDev.comp[component]) 
}

func (s *Sim) GetMinM(component int) {
	Send(s.getMinM(2 - component)) // translate to ZYX

}
func (s *Sim) getMinM(component int) float32 {
	s.init()
	return s.devmin.Reduce(s.mDev.comp[component]) 
}

// Gets the maximum torque expressed in gamma*Ms, as set by the current solver.
func (s *Sim) GetMaxTorque() {
	s.init()
	Send(s.torque)
}


func (s* Sim) GetCorePos(){
	s.init()
	Send(s.corePos()[0])
	Send(s.corePos()[1])
}
