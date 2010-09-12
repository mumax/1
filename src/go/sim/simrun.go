package sim

import (
	"os"
)

// This file implements the methods for time stepping

// Run the simulation for a certain duration, specified in seconds
func (s *Sim) Run(time float64) {
	Debug("Running for", time, "s")
	time /= float64(s.UnitTime())
	s.init()
	stop := s.time + time

	for s.time < stop {

		// save output if so scheduled
		for _, out := range s.outschedule {
			if out.NeedSave(float(s.time) * s.UnitTime()) { // output entries want SI units
				// assure the local copy of m is up to date and increment the autosave counter if neccesary
				s.assureMUpToDate()
				// save
				out.Save(s)
				// here it should say out.sinceoutput = s.time * s.unittime, not in each output struct...
			}
		}

		updateDashboard(s)

		// step
		Debugvv("Step", s.steps)
		s.Start("Step")
		s.Step()
		s.steps++
		s.time += float64(s.dt)
		s.mUpToDate = false
		s.Stop("Step")

	}
	s.PrintTimer(os.Stdout)
	//does not invalidate
}

// INTERNAL
// Assures the local copy of m is up to date with that on the device
// If necessary, it will be copied from the device and autosaveIdx will be incremented
func (s *Sim) assureMUpToDate() {
	s.init()
	if !s.mUpToDate {
		Debugvv("Copying m from device to local memory")
		TensorCopyFrom(s.mDev, s.mLocal)
		s.autosaveIdx++
		s.mUpToDate = true
	}
}
