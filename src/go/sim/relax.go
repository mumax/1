//  Copyright 2010  Arne Vansteenkiste
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package sim

import ()


type Relax struct {
	sim *Sim
	Reductor
	m1est     *DevTensor
	t0        *DevTensor
	maxtorque float32
}

// sets the maximum residual torque
func (r *Relax) MaxTorque(maxtorque float32) {
	r.maxtorque = maxtorque
}

func NewRelax(s *Sim) *Relax {
	r := new(Relax)
	r.sim = s
	r.Reductor.InitMaxAbs(s.Backend, prod(s.size4D[0:]))
	r.m1est = NewTensor(s.Backend, Size4D(s.size[0:]))
	r.t0 = NewTensor(s.Backend, Size4D(s.size[0:]))
	return r
}

const (
	RELAX_START_DT     = 1e-5
	DEFAULT_MAX_TORQUE = 1e-5
)

// One relaxation step.
// sets sim.torque to the maximum torque so the driver can decide whether or not to stop.
func (r *Relax) RelaxStep()  {

	m := r.sim.mDev
	m1est := r.m1est
	h := r.sim.hDev
	sim := r.sim
	maxError := float32(1e-4)

	sim.calcHeff(m, h)
	sim.DeltaM(m, h, r.sim.dt)
	sim.torque = r.Reduce(h) / r.sim.dt // torque = delta M / delta t
	TensorCopyOn(h, r.t0)
	TensorCopyOn(m, m1est)
	sim.Add(m1est, r.t0)
	sim.Normalize(m1est) // Euler estimate

	sim.calcHeff(m1est, h)
	sim.DeltaM(m1est, h, r.sim.dt)
	tm1est := h
	t := tm1est
	sim.LinearCombination(t, r.t0, 0.5, 0.5)
	sim.Add(m, t)
	sim.Normalize(m) // Heun solution

	// Error estimate
	sim.MAdd(m1est, -1, m) // difference between Heun and Euler
	error := r.Reduce(m1est)

	// calculate new step
	factor := maxError / error
	// do not increase by time step by more than 100%
	if factor > 2. {
		factor = 2.
	}
	// do not decrease to less than 1%
	if factor < 0.01 {
		factor = 0.01
	}

	//sim.time += float64(sim.dt) // TODO: remove when output per step can be saved?
	sim.dt *= factor
}
