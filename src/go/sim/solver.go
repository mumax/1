package sim

import "strings"

type Solver interface {
	Step()
	SetDt(dt float)
}

func NewSolver(solvertype string, field *Field) Solver {
	solvertype = strings.ToLower(solvertype)
	switch solvertype {
	default:
		panic("Unknown solver type: " + solvertype + ". Options are: euler, semianal, heun.")
	case "euler":
		return &Euler{SolverState{0., field}}
	case "heun":
		return NewHeun(field)
	case "semianal":
		return &SemiAnal{SolverState{0., field}, 0} //0th order by default TODO: make selectable ("semianal0", "semianal1" ?)
	}
	panic("bug")
	return nil // never reached
}

// stores the common data for fixed-step solvers
type SolverState struct {
	Dt float
	*Field
}

func (s *SolverState) SetDt(dt float) {
	s.Dt = dt
}
