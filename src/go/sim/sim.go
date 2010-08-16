package sim

import (
	"fmt"
	// 	"strings"
	   "tensor"
)

// Stores a simulation state
// Here, all parameters are STILL IN SI UNITS
// when Sim.init() is called, a solver is initd with these values converted to internal units.
// We need to keep the originial SI values in case a parameter gets changed during the simulation and we need to re-initialize everything.
type Sim struct {

    backend Backend
    
	aexch float
	msat  float
	alpha float

	size     [3]int
	cellsize [3]float

  m *tensor.Tensor4
  
	dt     float
	time   float
	solver *Euler //TODO other types, embed

	savem       float
	autosaveIdx int

	hext [3]float


}

func New() *Sim {
	return NewSim()
}

func NewSim() *Sim {
	sim := new(Sim)
	sim.backend = GPU //the default TODO: check if GPU is present, use CPU otherwise
	sim.invalidate()
	return sim
}

// when a parmeter is changed, the simulation state is invalid until it gets (re-)initialized by init()
func (s *Sim) invalidate() {
	s.solver = nil
}

// when it returns false, init() needs to be called before running
func (s *Sim) isValid() bool {
	return s.solver != nil
}

// (re-)initialize the simulation tree, necessary before running
func (s *Sim) init() {
	if s.isValid() {
		return //no work to do
	}

  if s.m == nil{
    panic("m not set")
  }

	dev := s.backend

	mat := NewMaterial()
	mat.MSat = s.msat
	mat.AExch = s.aexch
	mat.Alpha = s.alpha

	size := s.size[0:]
	L := mat.UnitLength()
	cellsize := []float{s.cellsize[X] / L, s.cellsize[Y] / L, s.cellsize[Z] / L}

	magnet := NewMagnet(dev, mat, size, cellsize)

	dt := s.dt / mat.UnitTime()
	s.solver = NewEuler(dev, magnet, dt)

	B := s.solver.UnitField()
	s.solver.Hext = []float{s.hext[X] / B, s.hext[Y] / B, s.hext[Z] / B}

	fmt.Println(s.solver)

		TensorCopyTo(s.m, s.solver.M())
		s.solver.Normalize(s.solver.M())
		
	/*
		file := 0
		for i := 0; i < 100; i++ {
			TensorCopyFrom(solver.M(), m)
			fname := "m" + fmt.Sprintf("%06d", file) + ".t"
			file++
			tensor.WriteFile(fname, m)
			for j := 0; j < 100; j++ {
				solver.Step()
			}
		}

		solver.Dt = 0.01E-12 / mat.UnitTime()
		solver.Alpha = 0.02
	*/

}

func (s *Sim) AExch(a float) {
	s.aexch = a
	s.invalidate()
}

func (s *Sim) MSat(ms float) {
	s.msat = ms
	s.invalidate()
}

func (s *Sim) Alpha(a float) {
	s.alpha = a
	s.invalidate()
}

func (s *Sim) Size(x, y, z int) {
	s.size[X] = x
	s.size[Y] = y
	s.size[Z] = z
	s.invalidate()
}

func (s *Sim) CellSize(x, y, z float) {
	s.cellsize[X] = x
	s.cellsize[Y] = y
	s.cellsize[Z] = z
	s.invalidate()
}

func (s *Sim) Dt(t float) {
	s.dt = t
	s.invalidate()
}

func (s *Sim) Field(hx, hy, hz float) {
	s.hext[X] = hx
	s.hext[Y] = hy
	s.hext[Z] = hz
	s.invalidate()
}

func (s *Sim) Verbosity(level int) {
	Verbosity = level
	// does not invalidate
}

func (s *Sim) Run(time float) {

	s.init()
	stop := s.time + time
	sinceout := 0.

	for s.time < stop {

		s.solver.Step()
		s.time += s.dt
    sinceout += s.dt
    
	  if s.savem > 0 && sinceout >= s.savem {
      sinceout = 0.
      s.autosavem()
    }
	}
	//does not invalidate
}

func (s *Sim) autosavem() {
	s.autosaveIdx++ // we start at 1 to stress that m0 has not been saved
	TensorCopyFrom(s.solver.M(), s.m)
	fname := "m" + fmt.Sprintf("%06d", s.autosaveIdx) + ".t"
	tensor.WriteFile(fname, s.m)
}

func (s *Sim) AutosaveM(interval float){
  s.savem = interval
  //does not invalidate
}

func (s *Sim) Cpu() {
	s.backend = CPU
	s.invalidate()
}

func (s *Sim) Gpu() {
	s.backend = GPU
	s.invalidate()
}


func (s *Sim) Uniform(mx, my, mz float){
  s.ensure_m()
  Uniform(s.m, mx, my, mz)
}

func (s *Sim) ensure_m(){
  if s.m == nil{
    s.m = tensor.NewTensor4([]int{3, s.size[X], s.size[Y], s.size[Z]})
  }
}

// func (s *Sim) Backend(b string) {
//   b = strings.ToLower(b)
//   switch b{
//     default:
//       panic(fmt.Sprint("Unknown backend:", b))
//     case "cpu":
//       s.backend = CPU
//     case "gpu":
//       s.backend = GPU
//   }
//   s.invalidate()
// }
