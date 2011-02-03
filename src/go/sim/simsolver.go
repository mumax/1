//  Copyright 2010  Arne Vansteenkiste
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package sim

import ()

// Set the solver type: euler, heun, semianal, ...
func (s *Sim) SolverType(stype string) {
	s.input.solvertype = stype
	s.invalidate()
}


// Set the solver time step, defined in seconds
// TODO this should imply or require a fixed-step solver
func (s *Sim) Dt(t float32) {
	s.input.dt = t
	s.invalidate()
}


// Set the solver maximum time step, defined in seconds
func (s *Sim) MaxDt(dt float32) {
	s.input.maxDt = dt
	//s.invalidate()
}


// Set the solver maximum time step, defined in seconds
func (s *Sim) MinDt(dt float32) {
	s.input.minDt = dt
	//s.invalidate()
}


// Set the maximum "delta m" the solver can take.
// 0 means not used.
// Some solvers ignore this value
func (s *Sim) MaxDm(deltaM float32) {
	s.input.maxDm = deltaM
}

// Set the minimum "delta m" the solver can take.
// 0 means not used.
// Some solvers ignore this value
func (s *Sim) MinDm(deltaM float32) {
	s.input.minDm = deltaM
}


// Set the maximum "delta m" the solver can take.
// 0 means not used.
// Some solvers ignore this value
func (s *Sim) MaxError(errorPerStep float32) {
	s.maxError = errorPerStep
}
