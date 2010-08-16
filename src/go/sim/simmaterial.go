package sim

func (s *Sim) AExch(a float) {
	s.aexch = a
	s.invalidate()
}

func (s *Sim) MSat(ms float) {
	s.msat = ms
	s.invalidate()
}

func (s *Sim) Alpha(a float) {
	s.alpha = a
	s.invalidate()
}
