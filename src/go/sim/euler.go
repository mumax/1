//  Copyright 2010  Arne Vansteenkiste
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package sim

import ()


// 1st order Euler method
type Euler struct {
	*Sim
}

func (this *Euler) String() string {
	return "Euler"
}


func (this *Euler) Step() {
	m, h := this.mDev, this.h
	alpha, dt := this.alpha, this.dt

	// 	this.Normalize(this.m)
	this.calcHeff(m, h)
	this.DeltaM(m, h, alpha, dt/(1+alpha*alpha))
	deltaM := h // h is overwritten by deltaM

	this.Add(m, deltaM)
	this.Normalize(m)
}
