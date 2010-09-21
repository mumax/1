package sim

import ()


// 1st order Euler method
type AdaptiveEuler struct {
	*Sim
	Reductor
}


func (this *AdaptiveEuler) Step() {
	m, h := this.mDev, this.h
	alpha, dt := this.alpha, this.dt

	// 	this.Normalize(this.m)
	this.calcHeff(m, h)
	this.DeltaM(m, h, alpha, dt/(1+alpha*alpha))
	deltaM := h // h is overwritten by deltaM

	this.Add(m, deltaM)
	this.Normalize(m)
}

func (this *Euler) String() string {
	return "Adaptive Euler"
}
