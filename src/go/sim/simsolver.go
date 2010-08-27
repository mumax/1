package sim

import ()

func (s *Sim) SolverType(stype string) {
	s.solvertype = stype
	s.invalidate()
}
