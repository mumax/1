package sim

import "strings"

type Solver interface {
	Step()
}

func NewSolver(solvertype string, sim *Sim) Solver {
	solvertype = strings.ToLower(solvertype)
	switch solvertype {
	default:
		panic("Unknown solver type: " + solvertype + ". Options are: euler, semianal, heun.")
	case "euler":
		return &Euler{sim}
	case "heun":
		return NewHeun(sim)
// 	case "semianal":
// 		return &SemiAnal{SolverState{0., sim}, 0} //0th order by default TODO: make selectable ("semianal0", "semianal1" ?)
	}
	panic("bug")
	return nil // never reached
}
