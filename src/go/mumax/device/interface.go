//  This file is part of MuMax, a high-perfomrance micromagnetic simulator.
//  Copyright 2010  Arne Vansteenkiste
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package device

import ()

// The device.Interface makes an abstraction from a library with
// basic simulation functions for a specific computing device
// like a GPU or CPU (or possibly even a cluster).
//
// The interface specifies quite a number of simulation primitives
// like fft's, deltaM(), memory allocation... where all higher-level
// simulation functions can be derived from.
//
// Gpu is the primary implementation of the Device interface:
// eachs of its functions calls a corresponding C function that does
// the actual work with CUDA.
//
// The GPU implementation can be easily translated to a CPU alternative
// by just putting the CUDA kernels inside (openMP) for-loops instead of
// kernel launches. This straightforward translation is wrapped in
// Cpu
//
// The first layer of higher-level functions is implemented in device.go
//
// By convention, the methods in the Device interface are unsafe
// and therefore package private. They have safe, public wrappers
// derived methods device.go. This allows the safety checks to
// be implemented only once and not for each Device.
type Interface interface {

	// Returns the maximum number of threads on this device
	// CPU will return the number of processors,
	// GPU will return the maximum number of threads per block
	// init() does not need to be called before this function,
	// but may rather use it.
	maxthreads() int

	// Initiate the device.
	// threads sets the number of threads:
	// this is the number of hardware threads for CPU
	// or the number of threads per block for GPU
	// Options is currently not used but here to allow
	// additional fine-tuning in the future.
	init(threads, options int)

	//____________________________________________________________________ general purpose (use Backend safe wrappers)

	// adds b to a. N = length of a = length of b
	add(a, b uintptr, N int)

	// vector-constant multiply-add a[i] += cnst * b[i]
	madd(a uintptr, cnst float32, b uintptr, N int)

	// vector-vector multiply-add a[i] += b[i] * c[i]
	madd2(a, b, c uintptr, N int)

	// Adds a linear anisotropy contribution to h:
	// h_i += Sum_i k_ij * m_j
	// Used for edge corrections.
	addLinAnis(hx, hy, hz, mx, my, mz, kxx, kyy, kzz, kyz, kxz, kxy uintptr, N int)

	// adds the constant cnst to a. N = length of a
	addConstant(a uintptr, cnst float32, N int)

	// a = a * weightA + b * weightB
	linearCombination(a, b uintptr, weightA, weightB float32, N int)

	linearCombinationMany(result uintptr, vectors []uintptr, weights []float32, NElem int)

	// partial data reduction (operation = add, max, maxabs, ...)
	// input data size = N
	// output = partially reduced data, usually reduced further on CPU. size = blocks
	// patially reduce in "blocks" blocks, partial results in output. blocks = divUp(N, threadsPerBlock*2)
	// use "threads" threads per block: @warning must be < N
	// size "N" of input data, must be > threadsPerBlock
	reduce(operation int, input, output uintptr, buffer *float32, blocks, threads, N int) float32

	// normalizes a vector field. N = length of one component
	normalize(m uintptr, N int)

	// normalizes a vector field and multiplies with normMap. N = length of one component = length of normMap
	normalizeMap(m, normMap uintptr, N int)

	// Safe version: func (*Sim) DeltaM()
	// overwrites h with torque(m, h) * dtGilbert. N = length of one component
	deltaM(m, h uintptr, alpha, dtGilbert float32, N int)

	// Safe version: func (*Sim) DeltaM()
	// overwrites h with torque(m, h) * dtGilbert, inculding spin-transfer torque terms. size = of one component
	// dtGilb = dt / (1+alpha^2)
	// alpha = damping
	// beta = b(1+alpha*xi)
	// epsillon = b(xi-alpha)
	// b = µB / e * Ms (Bohr magneton, electron charge, saturation magnetization)
	// u = current density / (2*cell size)
	// here be dragons
	spintorqueDeltaM(m, h uintptr, alpha, beta, epsillon float32, u []float32, dtGilb float32, size []int)

	addLocalFields(m, h uintptr, Hext []float32, anisType int, anisK []float32, anisAxes []float32, N int)

	// Override the GPU stride, handy for debugging. -1 Means reset to the original GPU stride
	// TODO: get rid of? decide the stride by yourself instead of globally storing it?
	overrideStride(nFloats int)

	//____________________________________________________________________ tensor (safe wrappers in tensor.go)

	copyPadded(source, dest uintptr, sourceSize, destSize []int, direction int)

	// Allocates a float array of size components*nFloats on the Device.
	// uintptrs to each component are returned. 
	// Nevertheless the underlying storage is contiguous.
	//
	// Does not need to be initialized with zeros
	//
	// By convention, Device arrays are represented by a uintptr,
	// while host arrays are []float32's.
	// 
	// "components" is needed for a multi-GPU array. See multigpu.go
	newArray(components, nFloats int) []uintptr

	// Frees device memory allocated by newArray
	freeArray(ptr uintptr)

	// Copies nFloats to, on or from the device, depending on the direction flag (1, 2 or 3)
	memcpy(source, dest uintptr, nFloats, direction int)

	// Offset the array pointer by "index" float32s, useful for taking sub-arrays
	// TODO: on a multi-device this will not work
	//arrayOffset(array uintptr, index int) uintptr

	// Overwrite n float32s with zeros
	zero(data uintptr, nFloats int)

	//____________________________________________________________________ specialized (used in only one place)

	// N: number of vectors 
	semianalStep(min, mout, h uintptr, dt, alpha float32, N int)

	// In-place kernel multiplication (m gets overwritten by h).
	// The kernel is symmetric so at most 6 of the 9 components need to be passed (xx, yy, zz, yz, xz, xy).
	// Even less components may be needed when some are zero. 
	// The number of non-zero components defines "kerneltype" 
	// 3D   micromagnetic problem: kerneltype = 6 
	// 2D   micromagnetic problem: kerneltype = 4 (only 4 of the 9 components need to be passed: xx, yy, zz, yz).
	// 2.5D micromagnetic problem: kerneltype = 3 (only 3 of the 9 components need to be passed: yy, zz, yz).
	// (this is the typical situation for a infinitely thick 2D micromagnetic problem,
	// which has no demag effects in the out-of-plane direction)
	kernelMul(mx, my, mz, kxx, kyy, kzz, kyz, kxz, kxy uintptr, kerneltype, nRealNumbers int)

	// unsafe creation of C fftPlan
	newFFTPlan(dataSize, logicSize []int) uintptr

	// execute the fft
	fft(plan uintptr, in, out uintptr, direction int)

	freeFFTPlan(plan uintptr)

	//______________________________________________________________________________ already safe

	// The GPU stride in number of float32s (!)
	// Stride() int

	// Bytes allocated on the device
	// UsedMem() uint64

	// Print the GPU properties to stdout
	// TODO: return string
	// PrintProperties()

	// TimerPrintDetail()

	String() string
}
