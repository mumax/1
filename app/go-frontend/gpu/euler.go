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
  this.Exec(this.m, this.h)
}


