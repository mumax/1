//  This file is part of MuMax, a high-performance micromagnetic simulator
//  Copyright 2010  Arne Vansteenkiste
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package sim

// This file implements functionality to quickly relax the magnetization to the ground state.

import ()

func (s *Sim) Relax() {
	s.init()
	s.wantEnergy = true
	s.relaxstep()
	prevE := s.energy + 1 // to get in loop
	for s.energy <= prevE {
		prevE = s.energy
		s.relaxstep()
	}
}


func (s *Sim) relaxstep() {
	//backup_time := s.time
	s.step()
	//s.time = backup_time // HACK: during relax we want the time to stand still
	s.steps++
	s.mUpToDate = false
	updateDashboard(s)

	// TEMP	
	// save output if so scheduled
	for _, out := range s.outschedule {
		if out.NeedSave(float32(s.time) * s.UnitTime()) { // output entries want SI units
			// assure the local copy of m is up to date and increment the autosave counter if necessary
			s.assureOutputUpToDate()
			// save
			out.Save(s)
			// TODO here it should say out.sinceoutput = s.time * s.unittime, not in each output struct...
		}
	}
}


func isUnstable(t *[4]float32) bool {
	return sign(t[1]-t[0]) == sign(t[3]-t[2]) && sign(t[1]-t[0]) != sign(t[2]-t[1])
}

func sign(x float32) float32 {
	if x < 0 {
		return -1
	}
	return 1
}
