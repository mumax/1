package sim

// This file implements the methods for setting
// material parameters.

// Sets the exchange constant, defined in J/m
func (s *Sim) AExch(a float) {
	s.input.aexch = a
	s.invalidate()
}

// Sets the saturation magnetization, defined in A/m
func (s *Sim) MSat(ms float) {
	s.input.msat = ms
	s.invalidate()
}

// Sets the damping coefficient
func (s *Sim) Alpha(a float) {
	s.input.alpha = a
	s.invalidate()
}
