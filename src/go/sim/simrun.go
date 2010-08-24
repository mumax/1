package sim

// This file implements the methods for time stepping

// Set the solver time step, defined in seconds
// TODO this should imply or require a fixed-step solver
func (s *Sim) Dt(t float) {
	s.dt = t
	s.invalidate()
}

// Run the simulation for a certain duration, specified in seconds
func (s *Sim) Run(time float64) {

	s.init()
	stop := s.time + time

	for s.time < stop {
		// step
		s.Step()
		s.time += float64(s.dt)
		s.mUpToDate = false

		// save output if so scheduled
		for _, out := range s.outschedule {
			if out.NeedSave(float(s.time)) {
				// assure the local copy of m is up to date and increment the autosave counter if neccesary
				s.assureMUpToDate()
				// save
				out.Save(s)
			}
		}
	}
	//does not invalidate
}

// INTERNAL
// Assures the local copy of m is up to date with that on the device
// If necessary, it will be copied from the device and autosaveIdx will be incremented
func (s *Sim) assureMUpToDate() {
	s.init()
	if !s.mUpToDate {
		TensorCopyFrom(s.M(), s.m)
		s.autosaveIdx++
		s.mUpToDate = true
	}
}
