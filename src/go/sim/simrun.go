package sim

import (
	"os"
)

// This file implements the methods for time stepping

// Set the solver time step, defined in seconds
// TODO this should imply or require a fixed-step solver
func (s *Sim) Dt(t float) {
	s.dt = t
	s.invalidate()
}

// Run the simulation for a certain duration, specified in seconds
func (s *Sim) Run(time float64) {
	Debug("Running for", time, "s")
	s.init()
	stop := s.time + time

	for s.time < stop {

		// save output if so scheduled
		for _, out := range s.outschedule {
			if out.NeedSave(float(s.time)) {
				// assure the local copy of m is up to date and increment the autosave counter if neccesary
				s.assureMUpToDate()
				// save
				out.Save(s)
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
		Debugv("Copying m from device to local memory")
		TensorCopyFrom(s.mDev, s.mLocal)
		s.autosaveIdx++
		s.mUpToDate = true
	}
}
