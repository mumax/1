//  Copyright 2010  Arne Vansteenkiste
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package sim

import ()


type AdaptiveHeun struct {
	*Sim
	m1est *DevTensor
	t0    *DevTensor
	Reductor
}


func NewAdaptiveHeun(sim *Sim) *AdaptiveHeun {
	this := new(AdaptiveHeun)
	this.Sim = sim
	this.m1est = NewTensor(sim.backend, Size4D(sim.size[0:]))
	this.t0 = NewTensor(sim.backend, Size4D(sim.size[0:]))
	this.Reductor.InitMaxAbs(this.backend, prod(sim.size4D[0:]))
	// There has to be an initial dt set so we can start
	if this.dt == 0. {
		this.dt = 0.001 // initial dt guess (internal units)
	}
	return this
}


func (s *AdaptiveHeun) Step() {

	gilbertDt := s.dt / (1 + s.alpha*s.alpha)
	m := s.mDev
	m1est := s.m1est

	s.calcHeff(m, s.h)
	s.DeltaM(m, s.h, s.alpha, gilbertDt)
	TensorCopyOn(s.h, s.t0)
	TensorCopyOn(m, m1est)
	s.Add(m1est, s.t0)
	s.Normalize(m1est) // euler estimate

	s.calcHeff(s.m1est, s.h)
	s.DeltaM(s.m1est, s.h, s.alpha, gilbertDt)
	tm1est := s.h
	t := tm1est
	s.LinearCombination(t, s.t0, 0.5, 0.5)
	s.Add(m, t)
	s.Normalize(m) // heun solution

	if s.maxError != 0. {
		s.MAdd(m1est, -1, m) // difference between heun and euler
		error := s.Reduce(m1est)
		s.stepError = error
		// TODO if error is too large, undo the step

		// calculate new step
		s.dt = s.dt * s.maxError / error
	}
}


func (this *AdaptiveHeun) String() string {
	return "Adaptive Heun"
}
