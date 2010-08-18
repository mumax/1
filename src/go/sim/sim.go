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
  outputdir   string
  
	hext [3]float
}

func New() *Sim {
	return NewSim()
}

func NewSim() *Sim {
	sim := new(Sim)
	sim.backend = GPU //the default TODO: check if GPU is present, use CPU otherwise
	sim.outputdir = "."
	sim.invalidate()  //just to make sure we will init()
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

	if s.m == nil {
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

}


func (s *Sim) Verbosity(level int) {
	Verbosity = level
	// does not invalidate
}
