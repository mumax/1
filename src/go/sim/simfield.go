package sim

func (s *Sim) Field(hx, hy, hz float) {
	s.hext[X] = hx
	s.hext[Y] = hy
	s.hext[Z] = hz
	s.invalidate()
}
