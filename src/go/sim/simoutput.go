package sim

import (
	"fmt"
	"tensor"
)

func (s *Sim) autosavem() {
	s.autosaveIdx++ // we start at 1 to stress that m0 has not been saved
	TensorCopyFrom(s.solver.M(), s.m)
	fname := "m" + fmt.Sprintf("%06d", s.autosaveIdx) + ".t"
	tensor.WriteFile(fname, s.m)
}

func (s *Sim) AutosaveM(interval float) {
	s.savem = interval
	//does not invalidate
}
