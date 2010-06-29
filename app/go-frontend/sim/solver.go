package sim

import(
  "tensor"
)

type Solver struct{
  m, h *Tensor
  alpha, dt float
  Field
}


func(s *Solver) Torque(m, h *Tensor, alpha, dtGilbert float){
  assert(len(m.size) == 4)
  assert(tensor.EqualSize(m.size, h.size))
  
  N := m.size[1] * m.size[2] * m.size[3]
  s.torque(m.data, h.data, alpha, dtGilbert, N)
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

