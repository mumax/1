package sim

// This file implements the methods for defining
// the applied magnetic field.

// Apply a static field defined in Tesla
func (s *Sim) AppliedField(hx, hy, hz float) {
	s.hext[X] = hx
	s.hext[Y] = hy
	s.hext[Z] = hz
	s.invalidate()
}
