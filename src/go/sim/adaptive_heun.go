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
	this.m1est = NewTensor(sim.Backend, Size4D(sim.size[0:]))
	this.t0 = NewTensor(sim.Backend, Size4D(sim.size[0:]))
	this.Reductor.InitMaxAbs(sim.Backend, prod(sim.size4D[0:]))
	// There has to be an initial dt set so we can start
	if this.dt == 0. {
		this.dt = 0.00001 // initial dt guess (internal units)
	}
	return this
}


func (s *AdaptiveHeun) step() {

	m := s.mDev
	m1est := s.m1est

	s.calcHeff(m, s.hDev)
	s.DeltaM(m, s.hDev, s.dt)
	TensorCopyOn(s.hDev, s.t0)
	TensorCopyOn(m, m1est)
	s.Add(m1est, s.t0)
	s.Normalize(m1est) // Euler estimate

	s.calcHeff(s.m1est, s.hDev)
	s.DeltaM(s.m1est, s.hDev, s.dt)
	tm1est := s.hDev
	t := tm1est
	s.LinearCombination(t, s.t0, 0.5, 0.5)
	s.Add(m, t)
	s.Normalize(m) // Heun solution

	s.time += float64(s.dt)

	if s.input.maxError != 0. {
		s.MAdd(m1est, -1, m) // difference between Heun and Euler
		error := s.Reduce(m1est)
		s.stepError = error
		// TODO if error is too large, undo the step

		// calculate new step
		factor := s.input.maxError / error
		// do not increase by time step by more than 100%
		if factor > 2. {
			factor = 2.
		}
		// do not decrease to less than 1%
		if factor < 0.01 {
			factor = 0.01
		}

		s.dt = s.dt * factor
	}

}


func (this *AdaptiveHeun) String() string {
	return "Adaptive Heun"
}
