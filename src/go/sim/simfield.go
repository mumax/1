package sim

// This file implements the methods for defining
// the applied magnetic field.

// Apply a static field defined in Tesla
func (s *Sim) StaticField(hx, hy, hz float) {
  s.init()
	B := s.UnitField()
	s.AppliedField = &staticField{[3]float{hx / B, hy / B, hz / B}}
	//does not invalidate
}

type staticField struct {
	b [3]float
}

func (field *staticField) GetAppliedField(time float64) [3]float {
	return field.b
}

// Control the accuracy of the demag kernel.
// 2^accuracy points are used to integrate the field.
// A high value is accurate and slows down (only) the initialization.
func (s *Sim) DemagAccuracy(accuracy int) {
	Debugv("Demag accuracy:", accuracy)
	s.input.demag_accuracy = accuracy
	s.invalidate()
}


// Calculates the effective field of m and stores it in h
func (s *Sim) calcHeff(m, h *DevTensor) {
	// (1) Self-magnetostatic field
	// The convolution may include the exchange field
	s.Convolve(m, h)

	// (2) Add the externally applied field
	if s.AppliedField != nil {
		s.hext = s.GetAppliedField(s.time)
		for i := range s.hComp {
			s.AddConstant(s.hComp[i], s.hext[i])
		}
	}
}
