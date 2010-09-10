package sim

import ()


type Heun struct {
  *Sim
	m1est *DevTensor
	t0    *DevTensor
}


func NewHeun(f *Sim) *Heun {
	this := new(Heun)
	this.Sim = f
	this.m1est = NewTensor(f.Backend, Size4D(f.size[0:]))
	this.t0 = NewTensor(f.Backend, Size4D(f.size[0:]))
	return this
}


func (s *Heun) Step() {
	Debugvv("Heun.Step()")
	gilbertDt := s.dt / (1 + s.alpha*s.alpha)
	m := s.mDev
	m1est := s.m1est

	s.CalcHeff(m, s.h)
	s.DeltaM(m, s.h, s.alpha, gilbertDt)
	TensorCopyOn(s.h, s.t0)
	TensorCopyOn(m, m1est)
	s.Add(m1est, s.t0)
	s.Normalize(m1est)

	// debug: euler
	//     TensorCopyOn(m1est, m) // estimator is ok

	s.CalcHeff(s.m1est, s.h)
	s.DeltaM(s.m1est, s.h, s.alpha, gilbertDt)
	tm1est := s.h
	t := tm1est
	s.LinearCombination(t, s.t0, 0.5, 0.5)
	s.Add(m, t)

	s.Normalize(m)

	// 	// backup m
	// 	TensorCopyOn(thim, thim0)
	//
	// 	// euler step for m0
	// 	h0 := this.torque0
	// 	this.CalcHeff(thim0, h0)
	// 	this.DeltaM(thim0, this.torque0, this.Alpha, gilbertDt)
	// 	this.Add(thim0, this.torque0)
	// 	this.Normalize(thim0)
	// 	m1 := thim0
	//
	// 	// field after euler step
	// 	// todo need to update the time here, for time-dependent fields etc
	// 	torque1 := this.h
	// 	this.CalcHeff(m1, this.h)
	// 	this.DeltaM(m1, torque1, this.Alpha, gilbertDt)
	//
	// 	// combine deltaM of beginning and end of interval
	// 	this.LinearCombination(torque1, this.torque0, 0.5, 0.5)
	// 	this.Add(thim, torque1)
	// 	this.Normalize(thim)
}


func (this *Heun) String() string {
	return "Heun"
}
