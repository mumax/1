package gpu

import(
)

type Euler struct{
  Solver
  
}

func NewEuler(field *Field, dt, alpha float){
  euler := new(Euler)
  euler.dt = dt
}

func (this *Euler) Step(){
  m, h := this.m, this.h
  alpha, dt := this.alpha, this.dt
  
  this.Convolve(m, h)
  Torque(m, h, alpha, dt/(1+alpha*alpha))
  torque := h

  EulerStage(m, torque)
}


