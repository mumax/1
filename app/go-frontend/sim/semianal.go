package sim

import(
)


type SemiAnal struct{
  Solver
}


func NewSemiAnal(dev Backend, mag *Magnet, dt float) *SemiAnal{
  this := new(SemiAnal)
  this.Solver = *NewSolver(dev, mag)
  this.dt = dt
  return this
}


func (this *SemiAnal) Step(){
  Debugvv( "SemiAnal.Step()" )
  m, h := this.m, this.h
  alpha, dt := this.Alpha, this.dt

  this.Normalize(m)
  this.CalcHeff(this.m, this.h)
  this.semianalStep(m.data, h.data, dt, alpha, this.NSpins())
}


func(this *SemiAnal) String() string{
  return "SemiAnal" + this.Solver.String() + "--\n"
}
