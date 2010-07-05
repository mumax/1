package sim

import(
)


type SemiAnal struct{
  dt float
  Field
}


func NewSemiAnal(dev Backend, mag *Magnet, dt float) *SemiAnal{
  return &SemiAnal{dt, *NewField(dev, mag)}
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
  return "SemiAnal" + this.Field.String() + "--\n"
}
