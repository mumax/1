package sim

// import ()
//
//
// type SemiAnal struct {
// 	SolverState
// }
//
//
// func NewSemiAnal(dev Backend, mag *Magnet, Dt float) *SemiAnal {
// 	this := new(SemiAnal)
// 	this.Dt = Dt
// 	this.Field = *NewField(dev, mag)
// 	return this
// }
//
//
// func (this *SemiAnal) Step() {
// 	Debugvv("SemiAnal.Step()")
// 	m, h := this.m, this.h
// 	alpha, Dt := this.Alpha, this.Dt
//
// 	this.Normalize(m)
// 	this.CalcHeff(this.m, this.h)
// 	this.semianalStep(m.data, h.data, Dt, alpha, this.NSpins())
// }
//
//
// func (this *SemiAnal) String() string {
// 	return "SemiAnal" + this.Field.String() + "--\n"
// }
