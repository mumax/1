package sim

import (
	"tensor"
)


// This file contains functions to calculate the torque (dm/dt)
// and DeltaM


// calculates torque * dt, overwrites h with the result
func (dev *Sim) DeltaM(m, h *DevTensor, dt float32) {
  assert(len(m.size) == 4)
  assert(tensor.EqualSize(m.size, h.size))
  N := m.size[1] * m.size[2] * m.size[3]
  alpha := dev.alpha
  dtGilbert := dt / (1+alpha*alpha)
  dev.deltaM(m.data, h.data, alpha, dtGilbert, N)
}


// calculates torque * dt, overwrites h with the result
// func (dev *Sim) SpintorqueDeltaM(m, h *DevTensor, alpha, dtGilbert float32) {
//   assert(len(m.size) == 4)
//   assert(tensor.EqualSize(m.size, h.size))
//   N := m.size[1] * m.size[2] * m.size[3]
//   dev.deltaM(m.data, h.data, alpha, dtGilbert, N)
// }



// calculates torque, overwrites h with the result
func (dev *Sim) Torque(m, h *DevTensor) {
	assert(len(m.size) == 4)
	assert(tensor.EqualSize(m.size, h.size))
	dev.DeltaM(m, h, 1.0) // we (ab)use DeltaM with dt=1.
}
