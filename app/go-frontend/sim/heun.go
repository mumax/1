package sim

import(
)


type Heun struct{
  m0, torque0 *Tensor
  Solver
}


func NewHeun(dev Backend, mag *Magnet, dt float) *Heun{
  this := new(Heun)
  this.m0 = NewTensor(dev, mag.Size())
  this.torque0 = NewTensor(dev, mag.Size())
  this.Solver = *NewSolver(dev, mag)
  this.dt = dt
  return this
}


func (this *Heun) Step(){
  Debugvv( "Heun.Step()" )

  this.Normalize(this.m)
  TensorCopyOn(this.m, this.m0)
  
  this.CalcHeff(this.m, this.torque0)
  this.Torque(this.m, this.torque0, 0.5 * this.dt/(1+alpha*alpha))
  this.EulerStage(m, this.torque0)
  this.Normalize(this.m)

  this.CalcHeff(this.m, this.h)
  this.Torque(this.m, this.h, this.dt/(1+alpha*alpha))
  this.EulerStage(m, )
  this.Normalize(this.m)
}


func(this *Heun) String() string{
  return "Heun" + this.Solver.String() + "--\n"
}
