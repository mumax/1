//  This file is part of MuMax, a high-performance micromagnetic simulator
//  Copyright 2010  Arne Vansteenkiste
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package sim

import (
	. "mumax/common"
	"mumax/tensor"
	"os"
	"math"
	"fmt"
	"rand"
)

// When a parameter is changed, the simulation state is invalidated until it gets (re-)initialized by init().
func (s *Sim) invalidate() {
	if s.IsValid() {
		// HACK: since re-initialization is buggy, we do not allow it for the moment.
		panic(InputErr("This parameter could only be set once."))
	}
	s.valid = false
}


// When it returns false, init() needs to be called before running.
func (s *Sim) IsValid() bool {
	return s.valid
}


// (Re-)initialize the simulation tree, necessary before running.
func (s *Sim) init() {
	if s.IsValid() {
		return
		//		panic(InputErr("This parameter could only be set once."))
	}
	s.Println("Initializing simulation state")

	// Material parameters control the units so they need to be set up first
	s.initMaterial()

	// Size must be set before memory allocation
	s.initGridSize()
	s.initCellSize()

	// Allocate memory, but only if needed
	s.initDevMem()
	s.initMLocal()

	// check if m has been initialized
	s.checkInitialM()

	// copy to GPU and normalize on the GPU, according to the normmap.
	s.Println("Copy m from local to device")
	TensorCopyTo(s.mLocal, s.mDev)

	// 	s.Normalize(s.mDev) // mysteriously crashes
	// then copy back to local so we can see the normalized initial state.
	// (so m0000000.tensor is normalized)
	// 	TensorCopyFrom(s.mDev, s.mLocal)

	// (4) Calculate kernel & set up convolution
	s.initConv()

	// Edge corrections
	s.initGeom()

	// (5) Time stepping
	s.initSolver()

	s.valid = true // we can start the real work now
	s.BeenValid = true
}


func (s *Sim) initMaterial() {
	s.Println("Initializing material parameters")
	s.mu0 = 4.0E-7 * math.Pi
	s.gamma0 = 2.211E5
	s.muB = 9.2740091523E-24
	s.e = 1.60217646E-19

	if s.input.msat == 0. {
		s.Errorln("Saturation magnetization should first be set. E.g. msat 800E3")
		os.Exit(-6)
	}
	s.mSat = s.input.msat

	if s.input.aexch == 0. {
		s.Errorln("Exchange constant should first be set. E.g. aexch 12E-13")
		os.Exit(-6)
	}
	s.aExch = s.input.aexch

	if s.alpha <= 0. {
		s.Warn("Damping parameter alpha =  ", s.alpha)
	}

	if len(s.anisKInt) < len(s.input.anisKSI) {
		s.anisKInt = make([]float32, len(s.input.anisKSI))
	}
	for i := range s.input.anisKSI {
		s.anisKInt[i] = s.input.anisKSI[i] / s.UnitEnergyDensity()
	}

	s.desc["msat"] = s.mSat
	s.desc["aexch"] = s.aExch
	s.desc["alpha"] = s.alpha
}


func (s *Sim) initGridSize() {
	s.size4D[0] = 3 // 3-component vectors
	for i := range s.size {
		s.size[i] = s.input.size[i]
		if !(s.size[i] > 0) {
			s.Errorln("The gridsize should be set first. E.g.: gridsize 32 32 4")
			os.Exit(-6)
		}
		s.size4D[i+1] = s.size[i]
	}
	s.size3D = s.size4D[1:]
	if s.size[Z] == 1 {
		panic(InputErr("For a 2D geometry, use (X, Y, 1) cells, not (1, X, Y)"))
	}
	s.avgNorm = float32(tensor.Prod(s.size3D))
}


func (s *Sim) initCellSize() {
	L := s.UnitLength()
	for i := range s.size {
		s.cellSize[i] = s.input.cellSize[i] / L
		if !(s.cellSize[i] > 0.) {
			s.Errorln("The cell size should be set first. E.g. cellsize 1E-9 1E-9 1E-9")
			os.Exit(-6)
		}
	}
}


func (s *Sim) initDevMem() {
	// Free previous memory only if it has the wrong size
	if s.mDev != nil && !tensor.EqualSize(s.mDev.Size(), s.size4D[0:]) {
		s.Println("Freeing unused device memory")
		s.mDev.Free()
		s.hDev.Free()
		s.mDev = nil
		s.hDev = nil
	}

	if s.mDev == nil {
		s.Println("Allocating device memory " + fmt.Sprint(s.size4D))
		s.mDev = NewTensor(s.Backend, s.size4D[0:])
		s.hDev = NewTensor(s.Backend, s.size4D[0:])
	}

	s.initReductors()
}


func (s *Sim) initReductors() {
	//TODO: free the previous ones, preferentially in reductor.Init()
	N := Len(s.size3D)
	s.devsum.InitSum(s.Backend, N)
	s.devmaxabs.InitMaxAbs(s.Backend, N)
	s.devmax.InitMax(s.Backend, N)
	s.devmin.InitMin(s.Backend, N)
}


func (s *Sim) initMLocal() {
	s.initGridSize()
	if s.mLocal == nil {
		s.Println("Allocating local memory " + fmt.Sprint(s.size4D))
		s.mLocal = tensor.NewT4(s.size4D[0:])
		s.hLocal = tensor.NewT4(s.size4D[0:])
	} else {
		if !tensor.EqualSize(s.mLocal.Size(), Size4D(s.input.size[0:])) {
			s.Println("Resampling magnetization from ", s.mLocal.Size(), " to ", Size4D(s.input.size[0:]))
			s.mLocal = resample4(s.mLocal, Size4D(s.input.size[0:]))
			s.hLocal = tensor.NewT4(s.size4D[0:])
		}
	}

}


// checks if the initial magnetization makes sense
// TODO: m=0,0,0 should not be a warning when the corresponding normMap is zero as well.
func (s *Sim) checkInitialM() {
	// All zeros: not initialized: error
	list := s.mLocal.List()
	ok := false
	for _, m := range list {
		if m != 0. {
			ok = true
		}
	}
	if !ok {
		s.Warn("Initial magnetization was not set")
		panic(InputErr("Initial magnetization was not set"))
	}

	// Some zeros: may or may not be a mistake,
	// warn and set to random.
	warned := false
	m := s.mLocal.Array()
	for i := range m[0] {
		for j := range m[0][i] {
			for k := range m[0][i][j] {
				if m[X][i][j][k] == 0. && m[Y][i][j][k] == 0. && m[Z][i][j][k] == 0. {
					if !warned {
						s.Warn("Some initial magnetization vectors were zero, set to a random value")
						warned = true
					}
					m[X][i][j][k], m[Y][i][j][k], m[Z][i][j][k] = rand.Float32()-0.5, rand.Float32()-0.5, rand.Float32()-0.5
				}
			}
		}
	}
}


func (s *Sim) initConv() {
	s.paddedsize = padSize(s.size[:], s.input.periodic[:])

	s.Println("Calculating kernel (may take a moment)") // --- In fact, it takes 3 moments, one in each direction.
	// lookupKernel first searches the wisdom directory and only calculates the kernel when it's not cached yet.
	demag := s.LookupKernel(s.paddedsize, s.cellSize[0:], s.input.demag_accuracy, s.input.periodic[:])

	//	fmt.Println(demag)

	var exch []*tensor.T3
	switch s.input.exchType {
	default:
		panic(InputErr("Illegal exchange type: " + fmt.Sprint(s.input.exchType) + ". Options are: 0, 6, 12"))
	case 0: // no exchange
	case 6:
		exch = Exch6NgbrKernel(s.paddedsize, s.cellSize[0:])
	case 12:
		exch = Exch12NgbrKernel(s.paddedsize, s.cellSize[0:])
	//case 26:
		//exch = Exch26NgbrKernel(s.paddedsize, s.cellSize[0:])
	}

	// Add Exchange kernel to demag kernel
	if s.exchInConv && s.input.exchType != 0 {
		Println("Exchange included in convolution.")
		for i := range demag {
			if demag[i] != nil { // Unused components are nil
				D := demag[i].List()
				E := exch[i].List()
				for j := range D {
					D[j] += E[j]
				}
			}
		}
	} else {
		Println("Exchange separate from convolution.")
	}
	s.Conv = *NewConv(s.Backend, s.size[0:], demag)
}


func (s *Sim) initSolver() {
	if s.Solver == nil { // TODO: FOR DEBUG ONLY, SHOULD CHECK IF TYPE/SIZE IS STILL UP TO DATE
		s.Println("Initializing solver: ", s.input.solvertype)
		s.dt = s.input.dt / s.UnitTime()
		if s.dt == 0. {
			s.dt = DEFAULT_DT_INTERNAL
			s.Println("Using default initial dt: ", s.dt*s.UnitTime(), " s")
		}
		s.Solver = NewSolver(s.input.solvertype, s)
	}
}
