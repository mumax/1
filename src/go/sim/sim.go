//  This file is part of MuMax, a high-performance micromagnetic simulator
//  Copyright 2010  Arne Vansteenkiste
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

// Magnetic simulation package
package sim

import (
	. "mumax/common"
	"mumax/tensor"
	"mumax/omf"
	"time"
	"fmt"
	"os"
	"bufio"
)

const (
	DEFAULT_SOLVERTYPE        = "rk23"
	DEFAULT_MAXERROR          = 1e-4
	DEFAULT_DEMAG_ACCURACY    = 8
	DEFAULT_SPIN_POLARIZATION = 1
	DEFAULT_EXCH_TYPE         = 6
	DEFAULT_MIN_DM            = 0
	DEFAULT_DT_INTERNAL       = 1e-4
	DEFAULT_RELAX_MAX_TORQUE  = 1e-3
)

// Sim has an "input" member of type "Input".
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
// TODO: at least for mumax 1.x we need an even more strict
// separation between user input and the rest of the program state.
type Input struct {
	aexch          float32    // Exchange constant in SI units (J/m)
	msat           float32    // Saturation magnetization in SI units (A/m)
	size           [3]int     // Grid size in number of cells
	cellSize       [3]float32 // Cell size in SI units (m)
	maxCellSize    [3]float32 // Maximum cell size in SI units (m)
	partSize       [3]float32 // Total magnet size in SI units(m), = size * cellSize
	sizeSet        bool       // Input file may set only 2 of size, cellSize, partSize. The last one being calculated automatically. It is an error to set all 3 of them so we keep track of which is set by the user.
	cellSizeSet    bool
	maxCellSizeSet bool
	partSizeSet    bool
	demag_accuracy int
	dt             float32
	maxDt, minDt   float32
	maxDm, minDm   float32 // The min/max magnetization step ("delta m") to be taken by the solver. 0 means not used. May be ignored by certain solvers.
	solvertype     string
	j              [3]float32 // current density in A/m^2
	exchType       int        // exchange scheme: 6 or 26 neighbors
	maxError       float32    // The maximum error per step to be made by the solver. 0 means not used. May be ignored by certain solvers.
	periodic       [3]int     // Periodic boundary conditions? 0=no, >0=yes
	geom           Geom       // Shape of the magnet (has Inside(x,y,z) func)
	edgeCorr       int        // 0: no edge correction, >0: 2^edgecorr cell subsampling for edge corrections
	anisType       int        // Anisotropy type
	anisKSI        []float32  // Anisotropy constant(s), as many as needed
	anisAxes       []float32  // Anisotopy axes: ux,uy,uz for uniaxial, u1x,u1y,u1z,u2x,u2y,u2z for cubic
	kernelType     string     // Determines which kernel subprogram to use.
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
	input     Input // stores the original input parameters in SI units
	valid     bool  // false when an init() is needed, e.g. when the input parameters have changed and do not correspond to the simulation anymore
	BeenValid bool  // true if the sim has been valid at some point. used for idiot-proof input file handling (i.e. no "run" commands)

	mDev           *DevTensor // magnetization on the device (GPU), 4D tensor
	size3D         []int      //simulation grid size (without 3 as first element)
	hDev           *DevTensor // effective field OR TORQUE, on the device. This is first used as a buffer for H, which is then overwritten by the torque.
	mLocal, hLocal *tensor.T4 // a "local" copy of the magnetization (i.e., not on the GPU) use for I/O
	//mLocalLock     sync.RWMutex
	mUpToDate bool // Is mLocal up to date with mDev? If not, a copy form the device is needed before storing output.

	Conv             // Convolution plan for the magnetostatic field
	wisdomdir string // Absolute path of the kernel wisdom root directory

	Material // Stores material parameters and manages the internal units
	Mesh     // Stores the size of the simulation grid

	AppliedField            // returns the externally applied in function of time
	hextSI       [3]float32 // stores the externally applied field returned by AppliedField, in SI UNITS
	hextInt      []float32  // stores the externally applied field in internal units

	//relaxer   *Relax
	Solver         // Does the time stepping, can be euler, heun, ...
	time   float64 // The total time (internal units)
	dt     float32 // The time step (internal units). May be updated by adaptive-step solvers
	torque float32 // Buffer for the maximum torque. May or may not be updated by solvers. Used for output.

	//targetDt  float32 // Preferred time step for fixed dt solvers. May be overriden when delta m would violate minDm, maxDm
	stepError float32 // The actual error estimate of the last step. Not all solvers update this value.
	steps     int     // The total number of steps taken so far

	outschedule []Output       // List of things to output. Used by simoutput.go. TODO make this a Vector, clean up
	autosaveIdx int            // Unique identifier of output state. Updated each time output is saved.
	tabwriter   *omf.TabWriter // Writes the odt data table. Is defined here so the sim can close it.
	devsum      Reductor       // Reduces mx, my, mz (SUM) on the device, used to output the avarage magnetization
	devmaxabs   Reductor       // Reduces the torque (maxabs) on the device, used to output max dm/dt
	devmin      Reductor
	devmax      Reductor
	//  preciseStep  bool              // Should we cut the time step to save the output at the precise moment specified?

	silent    bool                   // Do not print anything to os.Stdout when silent == true, only to log file
	outputdir string                 // Where to save output files.
	out       *os.File               // Output log file
	desc      map[string]interface{} // Metadata to be added to headers of saved tensors
	starttime int64                  // Walltime when the simulation was started, seconds since unix epoch. Used by dashboard.go

	// Geometry
	normMap   *DevTensor   // Per-cell magnetization norm. nil means the norm is 1.0 everywhere. Stored on the device
	normLocal *tensor.T3   // local copy of devNorm
	avgNorm   float32      // "Total" magnetization norm over the magnet. Divide the sum of all magnetization vectors by this to get the average magnetization correctly, even when the geometry is not square.
	edgeKern  []*DevTensor // Per-cell self-kernel used for edge corrections (could also store some anisotropy types)

	// Benchmark info
	lastrunSteps            int
	lastrunWalltime         int64
	lastrunSimtime          float64
	LastrunStepsPerSecond   float64
	LastrunSimtimePerSecond float64

	anisKInt []float32 // Anisotropy constant(s), as many as needed, internal units
}


func New(outputdir string, backend *Backend) *Sim {
	return NewSim(outputdir, backend)
}


func NewSim(outputdir string, backend *Backend) *Sim {
	sim := new(Sim)
	sim.Backend = backend
	sim.Backend.init(*threads, parseDeviceOptions()) //TODO: should not use global variable threads here
	sim.starttime = time.Seconds()
	sim.outschedule = make([]Output, 50)[0:0]
	sim.mUpToDate = false
	sim.input.demag_accuracy = DEFAULT_DEMAG_ACCURACY
	sim.autosaveIdx = -1 // so we will start at 0 after the first increment
	sim.input.solvertype = DEFAULT_SOLVERTYPE
	sim.input.maxError = DEFAULT_MAXERROR
	//sim.minDm = DEFAULT_MIN_DM
	sim.input.exchType = DEFAULT_EXCH_TYPE
	sim.spinPol = DEFAULT_SPIN_POLARIZATION
	// We run the simulation with working directory = directory of input file
	// This is neccesary, e.g., when a sim deamon is run from a directory other
	// than the directory of the input file and files with relative paths are
	// read (e.g. "include file", "load file")
	workdir := ParentDir(outputdir)
	//Println("chdir ", workdir)
	os.Chdir(workdir)
	sim.outputDir(Filename(outputdir))
	sim.desc = make(map[string]interface{})
	sim.hextInt = make([]float32, 3)
	sim.initWriters()
	sim.input.anisKSI = []float32{0.} // even when not used these must be allocated
	sim.input.anisAxes = []float32{0.}
	sim.input.kernelType = "mumaxkern-gpu"
	sim.invalidate() //just to make sure we will init()
	return sim
}


const (
	FFTW_PATIENT_FLAG = 1 << iota
)

// makes the device options flag based on the CLI flags
// this flag is the bitwise OR of flags like FFTW_PATIENT, ...
func parseDeviceOptions() int {
	flags := 0
	if *patient {
		flags |= FFTW_PATIENT_FLAG
	}
	return flags
}


func (sim *Sim) Normalize(m *DevTensor) {
	assert(len(m.size) == 4)
	N := m.size[1] * m.size[2] * m.size[3]
	if sim.normMap == nil {
		sim.normalize(m.data, N)
	} else {
		assert(sim.normMap.size[0] == m.size[1] && sim.normMap.size[1] == m.size[2] && sim.normMap.size[2] == m.size[3])
		sim.normalizeMap(m.data, sim.normMap.data, N)
	}
}

// Appends some benchmarking information to a file:
// sizeX, sizeY, sizeZ, totalSize, stpesPerSecond, simtimePerRealtime, #solvertype
func (sim *Sim) SaveBenchmark(filename string) {
	needheader := !fileExists(filename)
	out, err := os.Open(filename, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0666)
	defer out.Close()
	if err != nil {
		panic(err)
	}
	out2 := bufio.NewWriter(out)
	defer out2.Flush()
	if needheader {
		fmt.Fprintln(out2, "# Benchmark info")
		fmt.Fprintln(out2, "# size_x\tsize_y\tsize_z\ttotal_size\tsteps_per_second\tsimulated_time/wall_time\t#solver_type\tdevice")
	}
	fmt.Fprint(out2, sim.size[Z], "\t", sim.size[Y], "\t", sim.size[X], "\t", sim.size[X]*sim.size[Y]*sim.size[Z], "\t")
	fmt.Fprintln(out2, sim.LastrunStepsPerSecond, "\t", sim.LastrunSimtimePerSecond, "\t# ", sim.input.solvertype, "\t", sim.Backend.String())
}


// Closes open output streams before terminating
// TODO: rename running to finished should be done here
func (sim *Sim) Close() {
	sim.out.Close()
	if sim.tabwriter != nil {
		sim.tabwriter.Close()
	}
}
