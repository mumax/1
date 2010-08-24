package sim

import ()


type SemiAnal struct {
	SolverState
}


func (this *SemiAnal) Step() {
	Debugvv("SemiAnal.Step()")
	m, h := this.m, this.h

	this.Normalize(m)
	this.CalcHeff(this.m, this.h)
	this.semianalStep(m.data, h.data, this.Dt, this.Alpha, this.NSpins())
}


func (this *SemiAnal) String() string {
	return "SemiAnal\n" + this.Field.String() + "--\n"
}
