package sim

import ()


type Heun struct {
	m1est *Tensor
	t0    *Tensor
	SolverState
}


func NewHeun(f *Field) *Heun {
	this := new(Heun)
	this.Field = f
	this.m1est = NewTensor(f.Backend, f.Size4D())
	this.t0 = NewTensor(f.Backend, f.Size4D())
	return this
}


func (s *Heun) Step() {
	Debugvv("Heun.Step()")
	gilbertDt := s.Dt / (1 + s.Alpha*s.Alpha)

	s.CalcHeff(s.m, s.h)
	s.DeltaM(s.m, s.h, s.Alpha, gilbertDt)
	TensorCopyOn(s.h, s.t0)
	TensorCopyOn(s.m, s.m1est)
	s.Add(s.m1est, s.t0)
	s.Normalize(s.m1est)

	// debug: euler
	//     TensorCopyOn(s.m1est, s.m) // estimator is ok

	s.CalcHeff(s.m1est, s.h)
	s.DeltaM(s.m1est, s.h, s.Alpha, gilbertDt)
	tm1est := s.h
	t := tm1est
	s.LinearCombination(t, s.t0, 0.5, 0.5)
	s.Add(s.m, t)

	s.Normalize(s.m)

	// 	// backup m
	// 	TensorCopyOn(this.m, this.m0)
	//
	// 	// euler step for m0
	// 	h0 := this.torque0
	// 	this.CalcHeff(this.m0, h0)
	// 	this.DeltaM(this.m0, this.torque0, this.Alpha, gilbertDt)
	// 	this.Add(this.m0, this.torque0)
	// 	this.Normalize(this.m0)
	// 	m1 := this.m0
	//
	// 	// field after euler step
	// 	// todo need to update the time here, for time-dependent fields etc
	// 	torque1 := this.h
	// 	this.CalcHeff(m1, this.h)
	// 	this.DeltaM(m1, torque1, this.Alpha, gilbertDt)
	//
	// 	// combine deltaM of beginning and end of interval
	// 	this.LinearCombination(torque1, this.torque0, 0.5, 0.5)
	// 	this.Add(this.m, torque1)
	// 	this.Normalize(this.m)
}


func (this *Heun) String() string {
	return "Heun\n" + this.Field.String() + "--\n"
}
