package sim

import ()

// Set the solver type: euler, heun, semianal, ...
func (s *Sim) SolverType(stype string) {
	s.input.solvertype = stype
	s.invalidate()
}


// Set the solver time step, defined in seconds
// TODO this should imply or require a fixed-step solver
func (s *Sim) Dt(t float) {
	s.input.dt = t
	s.invalidate()
}
