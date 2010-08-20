// Magnetic simulation package
package sim

import (
	"fmt"
	"tensor"
)

// The Sim struct stores a simulation state.
//
// Here, all parameters are STILL IN SI UNITS.
// When Sim.init() is called, a solver is initiated
// with these values converted to internal units.
// We need to keep the originial SI values in case a
// parameter gets changed during the simulation and
// we need to re-initialize everything.
//
// The Sim struct has a lot of exported methods.
// When an input file is processed, reflection is used
// to resolve commands in the file to methods and call them.
// (See sim*.go, refsh/)
// All these methods may be called repeatedly and in any
// order; we use decentralized initialization to make sure
// everything works out.
//
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

	outschedule []Output //TODO vector...
	autosaveIdx int
	outputdir   string
	mUpToDate   bool

	hext [3]float
}

func New() *Sim {
	return NewSim()
}

func NewSim() *Sim {
	sim := new(Sim)
	sim.backend = GPU //the default TODO: check if GPU is present, use CPU otherwise
	sim.outputdir = "."
	sim.outschedule = make([]Output, 50)[0:0]
	sim.mUpToDate = false
	sim.invalidate() //just to make sure we will init()
	return sim
}

// When a parmeter is changed, the simulation state is invalidated until it gets (re-)initialized by init().
func (s *Sim) invalidate() {
	s.solver = nil
}

// When it returns false, init() needs to be called before running.
func (s *Sim) isValid() bool {
	return s.solver != nil
}

// (Re-)initialize the simulation tree, necessary before running.
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

// Set how much debug info is printed. Level=0,1,2 or 3 for none, normal, verbose and very verbose.
func (s *Sim) Verbosity(level int) {
	Verbosity = level
	// does not invalidate
}
