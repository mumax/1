package sim

import(
  "tensor"
)

// Solver contains common code of all concrete solvers,
// who embed it
// TODO perhaps we should pull m, h up so they can stay when the solver is changed for another one
type Solver struct{
  m, h *Tensor
  dt float
  Field
}


func NewSolver(dev Backend, mag *Magnet) *Solver{
  solver := new(Solver)

  solver.m = NewTensor(dev, Size4D(mag.Size()))
  solver.h = NewTensor(dev, Size4D(mag.Size()))
  solver.Field = *NewField(dev, mag)

  return solver
}


func(s *Solver) String() string{
  return "Solver\n" + s.Field.String()
}

func(s *Solver) M() *Tensor{
  return s.m
}

// TODO do not pass alpha
func(s *Solver) Torque(m, h *Tensor, dtGilbert float){
  assert(len(m.size) == 4)
  assert(tensor.EqualSize(m.size, h.size))
  
  N := m.size[1] * m.size[2] * m.size[3]
  s.torque(m.data, h.data, s.Alpha, dtGilbert, N)
}


func(s *Solver) Normalize(m *Tensor){
  assert(len(m.size) == 4)

  N := m.size[1] * m.size[2] * m.size[3]
  s.normalize(m.data, N)
}


func(s *Solver) EulerStage(m, torque *Tensor){
  assert(len(m.size) == 4)
  assert(tensor.EqualSize(m.size, torque.size))

  N := m.size[1] * m.size[2] * m.size[3]
  s.eulerStage(m.data, torque.data, N)
 
}

