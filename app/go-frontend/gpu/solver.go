package gpu

import(
  "tensor"
)

type Solver struct{
  m, h *Tensor
  alpha, dt float
  Field
}


func Torque(m, h *Tensor, alpha, dtGilbert float){
  assert(len(m.size) == 4)
  assert(tensor.EqualSize(m.size, h.size))
  
  N := m.size[1] * m.size[2] * m.size[3]
  torque(m.data, h.data, alpha, dtGilbert, N)
}


func Normalize(m *Tensor){
  assert(len(m.size) == 4)

  N := m.size[1] * m.size[2] * m.size[3]
  normalize(m.data, N)
}


func EulerStage(m, torque *Tensor){
  assert(len(m.size) == 4)
  assert(tensor.EqualSize(m.size, torque.size))

  N := m.size[1] * m.size[2] * m.size[3]
  eulerStage(m.data, torque.data, N)
 
}

