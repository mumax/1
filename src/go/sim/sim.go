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
// TODO at least time and dt should be float64
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
	time   float64
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
	sim.invalidate()     //just to make sure we will init()
	sim.autosaveIdx = -1 // so we will start at 0 after the first increment
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
	s.ensure_m()

	dev := s.backend
	dev.Init()

	mat := NewMaterial()
	mat.MSat = s.msat
	mat.AExch = s.aexch
	mat.Alpha = s.alpha

	size := s.size[0:]
	L := mat.UnitLength()
	cellsize := []float{s.cellsize[X] / L, s.cellsize[Y] / L, s.cellsize[Z] / L}
	magnet := NewMagnet(dev, mat, size, cellsize)

	dt := s.dt / mat.UnitTime()
	s.solver = NewEuler(dev, magnet, dt) //TODO solver dt should be float64(?)

	B := s.solver.UnitField()
	s.solver.Hext = []float{s.hext[X] / B, s.hext[Y] / B, s.hext[Z] / B}

	fmt.Println(s.solver)

	if !tensor.EqualSize(s.m.Size(), s.solver.M().Size()) {
		s.m = resample(s.m, s.solver.M().size)
	}
	TensorCopyTo(s.m, s.solver.M())

	s.solver.Normalize(s.solver.M())
}

// Set how much debug info is printed. Level=0,1,2 or 3 for none, normal, verbose and very verbose.
func (s *Sim) Verbosity(level int) {
	Verbosity = level
	// does not invalidate
}

func resample(in *tensor.Tensor4, size2 []int) *tensor.Tensor4 {
	out := tensor.NewTensor4(size2)
	out_a := out.Array()
	in_a:= in.Array()
	size1 := in.Size()
	for c := range out_a {
		for i := range out_a[c] {
			i1 := (i * size1[1]) / size2[1]
			for j := range out_a[0][i] {
				j1 := (j * size1[2]) / size2[2]
				for k := range out_a[0][i][j] {
					k1 := (k * size1[3]) / size2[3]
					out_a[c][i][j][k] = in_a[c][i1][j1][k1]
				}
			}
		}
	}
	return out
}
