//  Copyright 2010  Arne Vansteenkiste
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package sim

import "strings"

type Solver interface {
	Step()
	// 	String() string
}

func NewSolver(solvertype string, sim *Sim) Solver {
	solvertype = strings.ToLower(solvertype)
	switch solvertype {
	default:
		panic("Unknown solver type: " + solvertype + ". Options are: rk1, rk2, rk12, rk3, rk23, rk4, rk45, rksemianal1")
	case "rk1":
		return NewRK1(sim)
	case "rk2":
		return NewRK2(sim)
	case "rk12":
		return NewRK12(sim)
	case "rk3":
		return NewRK3(sim)
	case "rk23":
		return NewRK23(sim)
	case "rk4":
		return NewRK4(sim)
	case "rk45", "rkdp":
		return NewRKDP(sim)
	case "rkck":
		return NewRKCK(sim)
	case "euler":
		return NewAdaptiveEuler(sim)
	case "fixedeuler":
		return NewEuler(sim)
	case "heun":
		return NewAdaptiveHeun(sim)
	case "semianal1":
		return NewSemiAnal1(sim)
  case "semianal2":
    return NewSemiAnal2(sim)
	}
	panic("bug")
	return nil // never reached
}
