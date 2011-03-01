//  Copyright 2010  Arne Vansteenkiste
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package sim

import ()


type Heun struct {
	*Sim
	m1est *DevTensor
	t0    *DevTensor
}


func NewHeun(f *Sim) *Heun {
	this := new(Heun)
	this.Sim = f
	this.m1est = NewTensor(f.Backend, Size4D(f.size[0:]))
	this.t0 = NewTensor(f.Backend, Size4D(f.size[0:]))
	return this
}


func (s *Heun) step() {
	m := s.mDev
	m1est := s.m1est

	s.calcHeff(m, s.hDev)
	s.DeltaM(m, s.hDev, s.dt)
	TensorCopyOn(s.hDev, s.t0)
	TensorCopyOn(m, m1est)
	s.Add(m1est, s.t0)
	s.Normalize(m1est)

	s.calcHeff(s.m1est, s.hDev)
	s.DeltaM(s.m1est, s.hDev, s.dt)
	tm1est := s.hDev
	t := tm1est
	s.LinearCombination(t, s.t0, 0.5, 0.5)
	s.Add(m, t)

	s.Normalize(m)
	s.time += float64(s.dt)
}


func (this *Heun) String() string {
	return "Heun"
}
