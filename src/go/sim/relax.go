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

	var startDm float32 = 1e-2
	var minDm float32 = 1e-6
	var maxTorque float32 = 1e-5

	s.Println("Relaxing until torque < ", maxtorque)
	s.Normalize(s.mDev)
	s.mUpToDate = false

	backup_maxdm := s.input.maxDm

	dm := startDm
	s.torque = 2 * maxTorque // to get in loop
	for s.torque > maxTorque {
		torque := [4]float32{1, 1, 1, 1}
		s.input.maxDm = dm
		s.input.minDm = dm
		// Take a few steps first
		for i := 0; i < 10; i++ {
			s.relaxstep()
			torque[0], torque[1], torque[2], torque[3] = s.torque, torque[0], torque[1], torque[2]
		}
		for !isUnstable(&torque) {
			s.relaxstep()
			torque[0], torque[1], torque[2], torque[3] = s.torque, torque[0], torque[1], torque[2]
		    //dm *= 1.01
			//s.input.maxDm = dm
			//s.input.minDm = dm
		}
		if dm > minDm {
			dm *= 0.8
			s.Println("\nmax delta m:", dm, "\n")
		}
	}

	s.input.maxDm = backup_maxdm
	s.updateMLocal() // Even if no output was saved, mLocal should be up to date for a possible next relax()
	s.assureOutputUpToDate()
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
