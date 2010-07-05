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
  gilbertdt := this.dt/(1+this.Alpha*this.Alpha)
  
  this.Normalize(this.m)
  TensorCopyOn(this.m, this.m0)
  
  this.CalcHeff(this.m0, this.torque0)
  this.Torque(this.m0, this.torque0, 0.5 * gilbertdt)
  this.EulerStage(this.m0, this.torque0)
  this.Normalize(this.m0)

  this.CalcHeff(this.m0, this.h)
  this.Torque(this.m0, this.h, gilbertdt)
  this.linearCombination(this.h.data, this.torque0.data, 0.5, 0.5, this.NFloats())
  this.EulerStage(this.m, this.torque0)
  this.Normalize(this.m)
}


func(this *Heun) String() string{
  return "Heun" + this.Solver.String() + "--\n"
}
