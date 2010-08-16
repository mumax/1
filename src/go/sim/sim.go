package sim

import (
	"fmt"
	//   "tensor"
)

// Stores a simulation state
// Here, all parameters are STILL IN SI UNITS
// when Sim.init() is called, a solver is initd with these values converted to internal units.
// We need to keep the originial SI values in case a parameter gets changed during the simulation and we need to re-initialize everything.
type Sim struct {
	// material parameters
	aexch float
	msat  float
	alpha float

	// geometry
	size     [3]int
	cellsize [3]float

	// time stepping
	dt     float
	time   float
	solver *Euler //TODO other types

	//external field
	hext [3]float
	// backend
	backend int
}

func New() *Sim {
	return NewSim()
}

func NewSim() *Sim {
	sim := new(Sim)
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
		fmt.Println("valid")
		return //no work to do
	}
	fmt.Println("invalid")

	dev := GPU

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
    
	//	fmt.Println(s.solver)

	// 	m := tensor.NewTensorN(Size4D(magnet.Size()))
	// 	for i := range m.List() {
	// 		m.List()[i] = 1.
	// 	}
	// 	TensorCopyTo(m, solver.M())
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


func (s *Sim) Run(time float) {
	s.init()
	stop := s.time + time
	for s.time < stop {
		s.solver.Step()
		s.time += s.dt
	}
}


func (s *Sim) Backend(b string) {
	panic("todo")
}
