package sim

import (
	"tensor"
// 	"fmt"
)


// This file contains functions to calculate the torque (dm/dt)
// and DeltaM


// calculates torque * dt, overwrites h with the result
func (s *Sim) DeltaM(m, h *DevTensor, dt float32) {
  //TODO smart switch between torque/spintorque

  s.SpintorqueDeltaM(m, h, dt)
  
// 	assert(len(m.size) == 4)
// 	assert(tensor.EqualSize(m.size, h.size))
// 	N := m.size[1] * m.size[2] * m.size[3]
// 	alpha := s.alpha
// 	dtGilbert := dt / (1 + alpha*alpha)
// 	s.deltaM(m.data, h.data, alpha, dtGilbert, N)
}


// overwrites h with torque(m, h) * dtGilbert, inculding spin-transfer torque terms.
// dtGilb = dt / (1+alpha^2)
// alpha = damping
// beta = b(1+alpha*xi)
// epsillon = b(xi-alpha)
// b = µB / e * Ms (Bohr magneton, electron charge, saturation magnetization)
// u = current density / (2*cell size)
// here be dragons
func (s *Sim) SpintorqueDeltaM(m, h *DevTensor, dt float32) {
	assert(len(m.size) == 4)
	assert(tensor.EqualSize(m.size, h.size))

	alpha := s.alpha
	dtGilb := dt / (1 + alpha*alpha)

  muB := s.muB / s.UnitMoment()
  e := s.e / s.UnitCharge()
  P := s.spinPol
  xi := s.xi
  // Ms = 1
  
	b := P * muB / (e * 1 * (1+xi*xi))
	
	beta := b * (1 + alpha*xi)
	epsillon := b * (xi - alpha)

	u := [3]float32{}
	for i := range u {
		u[i] = 0.5 * (s.input.j[i] / s.UnitCurrentDensity()) / (s.cellSize[i])
	}
	//fmt.Println("alpha ", alpha, ", beta ", beta, ", epsillon ", epsillon)
	s.spintorqueDeltaM(m.data, h.data, alpha, beta, epsillon, u[:], dtGilb, m.size[1:]) // TODO: we need sim.size3D, sim.size4D to avoid slicing al the time.
}


// calculates torque, overwrites h with the result
func (s *Sim) Torque(m, h *DevTensor) {
	assert(len(m.size) == 4)
	assert(tensor.EqualSize(m.size, h.size))
	s.DeltaM(m, h, 1.0) // we (ab)use DeltaM with dt=1.
}
