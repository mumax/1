//  Copyright 2010  Arne Vansteenkiste
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package sim

import ()


// Euler method with (optional) adaptive step size.
// The step size is controlled by sim.maxDm,
// the maximum "delta m" to be taken by the solver.
// maxDm = 0 makes it a fixed-step Euler method.
// This implementation makes Euler obsolete.
//
type AdaptiveEuler struct {
	*Sim
	Reductor
}

func NewAdaptiveEuler(s *Sim) *AdaptiveEuler {
	e := new(AdaptiveEuler)
	e.Sim = s
	//assert(e.Backend != nil)
	// We use the "maximum norm" of the torque to set dt.
	// Using the Euclidian norm would also work.
	e.Reductor.InitMaxAbs(s.Backend, prod(s.size4D[0:]))
	//	if s.maxDm == 0. {
	//		s.maxDm = EULER_DEFAULT_MAXDM
	//	}
	return e
}

func (this *AdaptiveEuler) step() {
	m, h := this.mDev, this.hDev

	// 	this.Normalize(this.m)
	this.calcHeff(m, h)
	this.Torque(m, h)
	torque := h // h is overwritten by deltaM

	// only set an adaptive step if maxDm is defined.
	if this.input.maxDm != 0. {
		maxtorque := this.Reduce(torque)
		this.dt = this.input.maxDm / maxtorque
	}

	this.MAdd(m, this.dt, torque)
	this.Normalize(m)
	this.time += float64(this.dt)
}

func (this *AdaptiveEuler) String() string {
	return "Adaptive Euler"
}

// Default maximum delta m to be taken by adaptive euler solver.
const EULER_DEFAULT_MAXDM = 0.01
