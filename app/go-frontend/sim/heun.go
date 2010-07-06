package sim

import(
)


type Heun struct{
  m0, torque0 *Tensor
  TimeStep
}


func NewHeun(dev Backend, mag *Magnet, Dt float) *Heun{
  this := new(Heun)
  this.Field = *NewField(dev, mag)
  this.Dt = Dt
  this.m0 = NewTensor(dev, mag.Size4D())
  this.torque0 = NewTensor(dev, mag.Size4D())
  return this
}


func (this *Heun) Step(){
  Debugvv( "Heun.Step()" )
  gilbertDt := this.Dt/(1+this.Alpha*this.Alpha)
  
  this.Normalize(this.m)
  TensorCopyOn(this.m, this.m0)
  h0 := 
  this.CalcHeff(this.m0, this.torque0)
  this.DeltaM(this.m0, this.torque0, this.Alpha, 0.5 * gilbertDt)
  this.Add(this.m0, this.torque0)
  this.Normalize(this.m0)

  this.CalcHeff(this.m0, this.h)
  this.DeltaM(this.m0, this.h, this.Alpha, gilbertDt)
  this.LinearCombination(this.h, this.torque0, 0.5, 0.5)
  this.Add(this.m, this.h)sdghjghfdg
  this.Normalize(this.m)
}


func(this *Heun) String() string{
  return "Heun" + this.Field.String() + "--\n"
}
