//  Copyright 2010  Arne Vansteenkiste
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any
//  copyright notices and prominently state that you modified it, giving a relevant date.

package sim

import ()


type SemiAnal1 struct {
	*Sim
}


func NewSemiAnal1(sim *Sim) *SemiAnal1 {
	this := new(SemiAnal1)
	this.Sim = sim
	return this
}


func (s *SemiAnal1) Step() {
	m := s.mDev
	s.calcHeff(m, s.hDev)
	s.semianalStep(m.data, s.hDev.data, s.dt, s.alpha, 0, m.length/3)
	s.Normalize(m)
}


func (this *SemiAnal1) String() string {
	return "Semianlytical 1"
}
