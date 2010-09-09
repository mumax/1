package sim

// This file implements the methods for setting
// material parameters.

// Sets the exchange constant, defined in J/m
func (s *Sim) SetAExch(a float) {
	s.aexch = a
	s.invalidate()
}

// Sets the saturation magnetization, defined in A/m
func (s *Sim) SetMSat(ms float) {
	s.msat = ms
	s.invalidate()
}

// Sets the damping coefficient
func (s *Sim) SetAlpha(a float) {
	s.alpha = a
	s.invalidate()
}
