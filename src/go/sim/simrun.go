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
		mUpToDate := false

		for _,out := range s.outschedule{
      if out.NeedSave(s.time){
        // assure the local copy of m is up to date and increment the autosave counter if neccesary
        if(!mUpToDate){
          TensorCopyFrom(s.solver.M(), s.m)
          s.autosaveIdx++
          mUpToDate = true
        }
        // save
        out.Save(s)
      }
    }

	}
	//does not invalidate
}
