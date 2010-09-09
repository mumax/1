package sim

import "strings"

type Solver interface {
	Step()
	SetDt(dt float)
}

func NewSolver(solvertype string, sim *Sim) Solver {
	solvertype = strings.ToLower(solvertype)
	switch solvertype {
	default:
		panic("Unknown solver type: " + solvertype + ". Options are: euler, semianal, heun.")
	case "euler":
		return &Euler{SolverState{0., sim}}
	case "heun":
		return NewHeun(sim)
	case "semianal":
		return &SemiAnal{SolverState{0., sim}, 0} //0th order by default TODO: make selectable ("semianal0", "semianal1" ?)
	}
	panic("bug")
	return nil // never reached
}

// stores the common data for fixed-step solvers
type SolverState struct {
	Dt float
	*Sim
}

func (s *SolverState) SetDt(dt float) {
	s.Dt = dt
}
