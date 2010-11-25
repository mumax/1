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
		panic("Unknown solver type: " + solvertype + ". Options are: fixedeuler, euler, semianal1, heun.")
	case "euler":
		return NewAdaptiveEuler(sim)
	case "fixedeuler":
		return NewEuler(sim)

	case "heun":
		return NewAdaptiveHeun(sim)
	case "semianal1":
		return NewSemiAnal1(sim)
	}
	panic("bug")
	return nil // never reached
}
