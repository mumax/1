// Magnetic simulation package
package sim

import (
	"tensor"
	"time"
	"fmt"
)

// Sim has an "input" member of type Input.
//
// In this struct, all parameters are STILL IN SI UNITS.
// When Sim.init() is called, a solver is initiated
// with these values converted to internal units.
// We need to keep the originial SI values in case a
// parameter gets changed during the simulation and
// we need to re-initialize everything.
//
// This struct is not embedded in Sim but appears as
// a member "input" so that we have to write, e.g.,
// sim.input.dt to make clear it is not necessarily the
// same as sim.dt (which is in internal units)
//
type Input struct {
	aexch          float
	msat           float
	alpha          float
	size           [3]int
	cellSize       [3]float
	demag_accuracy int
	dt             float
	solvertype     string
}


// The Sim struct stores a simulation state.
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
	valid     bool

	input Input

	// what we want
	backend *Backend
	mLocal  *tensor.Tensor4

	// what we have
	Material
	Mesh
	Conv
	AppliedField //function
	hext         [3]float
	mDev, h      *DevTensor // on device
	mComp, hComp [3]*DevTensor
	Solver
	time  float64
	dt    float
	steps int

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
	sim.backend = GPU//nil //TODO: check if GPU is present, use CPU otherwise
	sim.outputdir = "."
	sim.outschedule = make([]Output, 50)[0:0]
	sim.mUpToDate = false
	sim.invalidate() //just to make sure we will init()
	sim.input.demag_accuracy = 8
	sim.autosaveIdx = -1 // so we will start at 0 after the first increment
	sim.starttime = time.Seconds()
	sim.valid = false
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
	Debugv("Initializing simulation state")
  
	dev := s.backend
	dev.InitBackend()
	assert(dev != nil)

	// (1) Material parameters control the units,
	// so they need to be set up first
	s.InitMaterial()
	s.mSat = s.input.msat
	s.aExch = s.input.aexch
	s.alpha = s.input.alpha

	// (2) Size must be set before memory allocation
	L := s.UnitLength()
	s.size4D[0] = 3 // 3-component vectors
	for i := range s.size {
		s.size[i] = s.input.size[i]
		assert(s.size[i] > 0)
		s.size4D[i+1] = s.size[i]
		s.cellSize[i] = s.input.cellSize[i] / L
		assert(s.cellSize[i] > 0.)
	}

	// (3) Allocate memory, but only if needed
	// Free previous memory only if it has the wrong size
  // Todo device should not have been changed
// 	if s.mDev != nil && !tensor.EqualSize(s.mDev.Size(), s.size4D[0:]) {
// 		// TODO: free
// 		s.mDev = nil
// 		s.h = nil
// 	}

// 	if s.mDev == nil {
		Debugv("Allocating device memory " + fmt.Sprint(s.size4D))
		s.mDev = NewTensor(dev, s.size4D[0:])
		s.h = NewTensor(dev, s.size4D[0:])
		s.mComp, s.hComp = [3]*DevTensor{}, [3]*DevTensor{}
		for i := range s.mComp {
			s.mComp[i] = s.mDev.Component(i)
			s.hComp[i] = s.h.Component(i)
		}
// 	}

    if s.mLocal == nil {
      Debugv("Allocating local memory " + fmt.Sprint(s.size4D))
      s.mLocal = tensor.NewTensor4(s.size4D[0:])
    }

    if !tensor.EqualSize(s.mLocal.Size(), s.mDev.Size()){
      s.mLocal = resample(s.mLocal, s.mDev.size)
    }

	// (3b) resize the previous magnetization state
// 	if !tensor.EqualSize(s.mLocal.Size(), s.mDev.Size()) {
// xxx
// 	}
	TensorCopyTo(s.mLocal, s.mDev)
// 	s.Normalize(s.mDev)

	// (4) Calculate kernel & set up convolution

	s.paddedsize = padSize(s.size[0:])

	Debugv("Calculating kernel")
	demag := FaceKernel6(s.paddedsize, s.cellSize[0:], s.input.demag_accuracy)
	exch := Exch6NgbrKernel(s.paddedsize, s.cellSize[0:])
	// Add Exchange kernel to demag kernel
	for i := range demag {
		D := demag[i].List()
		E := exch[i].List()
		for j := range D {
			D[j] += E[j]
		}
	}
	s.Conv = *NewConv(dev, s.size[0:], demag)

	//  B := s.UnitField()
	//  s.Hext = []float{s.hext[X] / B, s.hext[Y] / B, s.hext[Z] / B}

	// (5) Time stepping
	s.dt = s.input.dt / s.UnitTime()
	s.Solver = NewSolver(s.input.solvertype, s)

  s.valid = true

//   fmt.Println(s)
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
