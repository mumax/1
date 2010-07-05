package sim

import(
)


type Heun struct{
  m0, torque0 *Tensor
  dt float
  Field
}


func NewHeun(dev Backend, mag *Magnet, dt float) *Heun{
  this := new(Heun)
  this.Field = *NewField(dev, mag)
  this.dt = dt
  this.m0 = NewTensor(dev, mag.Size4D())
  this.torque0 = NewTensor(dev, mag.Size4D())
  return this
}


func (this *Heun) Step(){
  Debugvv( "Heun.Step()" )
  gilbertdt := this.dt/(1+this.Alpha*this.Alpha)
  
  this.Normalize(this.m)
  TensorCopyOn(this.m, this.m0)
  
  this.CalcHeff(this.m0, this.torque0)
  this.DeltaM(this.m0, this.torque0, this.Alpha, 0.5 * gilbertdt)
  this.Add(this.m0, this.torque0)
  this.Normalize(this.m0)

  this.CalcHeff(this.m0, this.h)
  this.DeltaM(this.m0, this.h, this.Alpha, gilbertdt)
  this.LinearCombination(this.h, this.torque0, 0.5, 0.5)
  this.Add(this.m, this.torque0)
  this.Normalize(this.m)
}


func(this *Heun) String() string{
  return "Heun" + this.Field.String() + "--\n"
}
