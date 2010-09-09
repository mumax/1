// Magnetic simulation package
package sim

import (
	"tensor"
	"time"
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
// TODO order of initialization is too important in input file, should be more versatile
//
type Sim struct {

  starttime int64 // when the simulation was started, seconds since unix epoch -> dashboard
  valid bool
  
  // what we want
	backend *Backend

	aexch float
	msat  float
	alpha float

	size           [3]int
	cellsize       [3]float
	demag_accuracy int

	mLocal *tensor.Tensor4

	dt         float
	time       float64
	steps      int
	solvertype string

  
  
  // what we have
  Material
  Mesh
  Conv
  AppliedField //function
  hext [3]float
  mDev, h         *DevTensor // on device
  mComp, hComp [3]*DevTensor
  Solver

  // output
	outschedule []Output //TODO vector...
	autosaveIdx int
	outputdir   string
	mUpToDate   bool

}

func New() *Sim {
	return NewSim()
}

func NewSim() *Sim {
	sim := new(Sim)
	sim.backend = nil //TODO: check if GPU is present, use CPU otherwise
	sim.outputdir = "."
	sim.outschedule = make([]Output, 50)[0:0]
	sim.mUpToDate = false
	sim.invalidate() //just to make sure we will init()
	sim.demag_accuracy = 8
	sim.autosaveIdx = -1 // so we will start at 0 after the first increment
	sim.starttime = time.Seconds()
	return sim
}

// When a parmeter is changed, the simulation state is invalidated until it gets (re-)initialized by init().
func (s *Sim) invalidate() {
	if s.isValid() {
		Debugv("Simulation state invalidated")
	}
	s.valid = false
}

// When it returns false, init() needs to be called before running.
func (s *Sim) isValid() bool {
	return s.valid
}

// (Re-)initialize the simulation tree, necessary before running.
func (s *Sim) init() {
	if s.isValid() {
		return //no work to do
	}
	Debugv("Re-initializing simulation state")

	s.ensure_m()

	dev := s.backend
	dev.Init()

	s.Material.Init()
// 	s.MSat = s.msat
// 	s.AExch = s.aexch
// 	s.Alpha = s.alpha

// 	size := s.size[0:]
//	L := s.UnitLength()
// 	cellsize := []float{s.cellsize[X] / L, s.cellsize[Y] / L, s.cellsize[Z] / L}
	//magnet := NewMagnet(dev, mat, size, cellsize)
	//s.Field = NewField(dev, magnet, s.demag_accuracy)

	dt := s.dt / s.UnitTime()
	s.Solver = NewSolver(s.solvertype, s) //NewEuler(dev, s.Field, dt) //TODO solver dt should be float64(?)
	s.Solver.SetDt(dt)

// 	B := s.UnitField()
// 	s.Hext = []float{s.hext[X] / B, s.hext[Y] / B, s.hext[Z] / B}

	if !tensor.EqualSize(s.mLocal.Size(), s.mDev.Size()) {
		s.mLocal = resample(s.mLocal, s.mDev.size)
	}
	TensorCopyTo(s.mLocal, s.mDev)

	s.Normalize(s.mDev)
}


// Calculates the effective field of m and stores it in h
func (s *Sim) CalcHeff(m, h *DevTensor) {
 
 s.Convolve(m, h)

//  if s.Hext != nil {
//      for i := range s.hComp {
//          f.AddConstant(f.hComp[i], f.Hext[i])
//      }
//  }
}

// Set how much debug info is printed. Level=0,1,2 or 3 for none, normal, verbose and very verbose.
func (s *Sim) Verbosity(level int) {
	Verbosity = level
	// does not invalidate
}

func resample(in *tensor.Tensor4, size2 []int) *tensor.Tensor4 {
	Debugv("Resampling magnetization from", in.Size(), "to", size2)
	out := tensor.NewTensor4(size2)
	out_a := out.Array()
	in_a := in.Array()
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
