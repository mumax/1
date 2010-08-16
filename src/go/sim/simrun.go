package sim

func (s *Sim) Dt(t float) {
	s.dt = t
	s.invalidate()
}


func (s *Sim) Run(time float) {

	s.init()
	stop := s.time + time
	sinceout := 0.

	for s.time < stop {

		s.solver.Step()
		s.time += s.dt
		sinceout += s.dt

		if s.savem > 0 && sinceout >= s.savem {
			sinceout = 0.
			s.autosavem()
		}
	}
	//does not invalidate
}
