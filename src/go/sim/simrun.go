package sim

func (s *Sim) Dt(t float) {
	s.dt = t
	s.invalidate()
}


func (s *Sim) Run(time float) {

	s.init()
	stop := s.time + time

	for s.time < stop {

		s.solver.Step()
		s.time += s.dt
		s.mUpToDate = false

		for _, out := range s.outschedule {
			if out.NeedSave(s.time) {
				// assure the local copy of m is up to date and increment the autosave counter if neccesary
				s.assureMUpToDate()
				// save
				out.Save(s)
			}
		}

	}
	//does not invalidate
}

// Assures the local copy of m is up to date with that on the device
// If necessary, it will be copied from the device and autosaveIdx will be incremented
func (s *Sim) assureMUpToDate() {
  s.init()
	if !s.mUpToDate {
		TensorCopyFrom(s.solver.M(), s.m)
		s.autosaveIdx++
		s.mUpToDate = true
	}
}
