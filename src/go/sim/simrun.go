//  Copyright 2010  Arne Vansteenkiste
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package sim

import (
	. "mumax/common"
	"fmt"
	"math"
	clock "time"
)

// This file implements the methods for time stepping

// Run the simulation for a certain duration, specified in seconds
func (s *Sim) Run(time float64) {
	s.init()
	s.Println("Running for ", time, "s")
	time /= float64(s.UnitTime())
	stop := s.time + time
	s.Normalize(s.mDev)
	s.mUpToDate = false

	s.start_benchmark()
	for s.time < stop {

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

		updateDashboard(s)

		// step
		s.step()
		s.steps++
		s.mUpToDate = false

		if math.IsNaN(s.time) || math.IsInf(s.time, 0) {
			panic("Time step = " + fmt.Sprint(s.dt))
		}
	}

	s.stop_benchmark()
	s.updateMLocal() // Even if no output was saved, mLocal should be up to date for a possible next relax()
	s.assureOutputUpToDate()
	//does not invalidate
}

// Take one time step
func (s *Sim) Step() {
	s.init()
	s.mUpToDate = false

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

	updateDashboard(s)

	s.step()
	s.steps++
	s.mUpToDate = false

	if math.IsNaN(s.time) || math.IsInf(s.time, 0) {
		panic("Time step = " + fmt.Sprint(s.dt))
	}
	//does not invalidate
}

// Takes n time steps
func (s *Sim) Steps(n int) {
	for i := 0; i < n; i++ {
		s.Step()
	}
}

var maxtorque float32 = DEFAULT_RELAX_MAX_TORQUE


// re-initialize benchmark data
func (s *Sim) start_benchmark() {
	s.lastrunSteps = s.steps
	s.lastrunWalltime = clock.Nanoseconds()
	s.lastrunSimtime = s.time
}


// update benchmark data
func (s *Sim) stop_benchmark() {
	runtime := float64(clock.Nanoseconds()-s.lastrunWalltime) / 1e9 // time of last run in seconds
	runsteps := s.steps - s.lastrunSteps
	simtime := (s.time - s.lastrunSimtime) * float64(s.UnitTime())
	s.LastrunStepsPerSecond = float64(runsteps) / runtime
	s.LastrunSimtimePerSecond = simtime / runtime
}


// INTERNAL
// Assures the local copy of m is up to date with that on the device
// If necessary, it will be copied from the device and autosaveIdx will be incremented
func (s *Sim) assureOutputUpToDate() {
	s.init()
	if !s.mUpToDate {
		// 		Debugvv("Copying m from device to local memory")
		TensorCopyFrom(s.mDev, s.mLocal)
		s.autosaveIdx++
		s.mUpToDate = true
	}
	s.desc["time"] = s.time * float64(s.UnitTime())
	s.desc["Bx"] = s.hextSI[Z]
	s.desc["By"] = s.hextSI[Y]
	s.desc["Bz"] = s.hextSI[X]
	s.desc["torque"] = s.torque
	s.desc["id"] = s.autosaveIdx
	s.desc["iteration"] = s.steps
}

// Copies mDev to mLocal.
// Necessary after each run(), relax(), ...
func (s *Sim) updateMLocal() {
	s.Println("Copy m from device to local")
	TensorCopyFrom(s.mDev, s.mLocal)
}
