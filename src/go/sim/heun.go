package sim
//
// import ()
//
//
// type Heun struct {
// 	m0, torque0 *Tensor
// 	SolverState
// }
//
//
// func NewHeun(dev Backend, mag *Magnet, Dt float) *Heun {
// 	this := new(Heun)
// 	this.Field = *NewField(dev, mag)
// 	this.Dt = Dt
// 	this.m0 = NewTensor(dev, mag.Size4D())
// 	this.torque0 = NewTensor(dev, mag.Size4D())
// 	return this
// }
//
//
// func (this *Heun) Step() {
// 	Debugvv("Heun.Step()")
// 	gilbertDt := this.Dt / (1 + this.Alpha*this.Alpha)
//
// 	// backup m
// 	TensorCopyOn(this.m, this.m0)
//
// 	// euler step for m0
// 	this.CalcHeff(this.m0, this.torque0)
// 	this.DeltaM(this.m0, this.torque0, this.Alpha, gilbertDt)
// 	this.Add(this.m0, this.torque0)
// 	this.Normalize(this.m0)
//
// 	// field after euler step
// 	torque1 := this.h
// 	this.CalcHeff(this.m0, torque1)
// 	this.DeltaM(this.m0, torque1, this.Alpha, gilbertDt)
//
// 	// combine deltaM of beginning and end of interval
// 	this.LinearCombination(torque1, this.torque0, 0.5, 0.5)
// 	this.Add(this.m, torque1)
// 	this.Normalize(this.m)
// }
//
//
// func (this *Heun) String() string {
// 	return "Heun" + this.Field.String() + "--\n"
// }
