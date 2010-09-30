//  Copyright 2010  Arne Vansteenkiste
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package sim

import ()

// This file implements the methods for time stepping

// Run the simulation for a certain duration, specified in seconds
func (s *Sim) Run(time float64) {
	s.init()
	s.Println("Running for ", time, "s")
	time /= float64(s.UnitTime())
	stop := s.time + time

	for s.time < stop {

		// save output if so scheduled
		for _, out := range s.outschedule {
			if out.NeedSave(float(s.time) * s.UnitTime()) { // output entries want SI units
				// assure the local copy of m is up to date and increment the autosave counter if neccesary
				s.assureMUpToDate()
				// save
				out.Save(s)
				// TODO here it should say out.sinceoutput = s.time * s.unittime, not in each output struct...
			}
		}

		updateDashboard(s)

		// step
		s.Start("Step")
		s.Step()
		s.steps++
		s.time += float64(s.dt)
		s.mUpToDate = false
		s.Stop("Step")

	}
	// 	s.PrintTimer(os.Stdout)
	//does not invalidate
}

// INTERNAL
// Assures the local copy of m is up to date with that on the device
// If necessary, it will be copied from the device and autosaveIdx will be incremented
func (s *Sim) assureMUpToDate() {
	s.init()
	if !s.mUpToDate {
		// 		Debugvv("Copying m from device to local memory")
		TensorCopyFrom(s.mDev, s.mLocal)
		s.autosaveIdx++
		s.mUpToDate = true
	}
}
