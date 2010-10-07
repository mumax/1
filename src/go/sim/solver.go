//  Copyright 2010  Arne Vansteenkiste
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

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
		return NewAdaptiveEuler(sim)
	case "heun":
		return NewAdaptiveHeun(sim)
		// 	case "semianal":
		// 		return &SemiAnal{SolverState{0., sim}, 0} //0th order by default TODO: make selectable ("semianal0", "semianal1" ?)
	}
	panic("bug")
	return nil // never reached
}
