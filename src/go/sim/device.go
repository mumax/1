//  Copyright 2010  Arne Vansteenkiste
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package sim

import ()

/**
 * The Device interface makes an abstraction from a library with
 * basic simulation functions for a specific computing device
 * like a GPU or CPU (or possibly even a cluster).
 *
 * The interface specifies quite a number of simulation primitives
 * like fft's, deltaM(), memory allocation... where all higher-level
 * simulation functions can be derived from.
 *
 * Gpu is the primary implementation of the Device interface:
 * eachs of its functions calls a corresponding C function that does
 * the actual work with CUDA.
 *
 * The GPU implementation can be easily translated to a CPU alternative
 * by just putting the CUDA kernels inside (openMP) for-loops instead of
 * kernel launches. This straightforward translation is wrapped in
 * Cpu
 *
 * The first layer of higher-level functions is provided by the Backend
 * struct, which embeds a Device. Backend does not need to know whether
 * it uses a gpu.Device or cpu.Device, and so the code for both is
 * identical from this point on.
 *
 * By convention, the methods in the Device interface are unsafe
 * and therefore package private. They have safe, public wrappers
 * derived methods in Backend. This allows the safety checks to
 * be implemented only once in Backend and not for each Device.
 * The few methods that are already safe are accessible through
 * Backend thanks to embedding.
 */
type Device interface {
	init()

	// selects a device when more than one is present
	// (typically used for multiple GPU's, not useful for CPU)
	setDevice(devid int)

	//____________________________________________________________________ general purpose (use Backend safe wrappers)

	// adds b to a. N = length of a = length of b
	add(a, b uintptr, N int)

	// vector multiply-add a[i] += cnst * b[i]
	madd(a uintptr, cnst float, b uintptr, N int)

	// adds the constant cnst to a. N = length of a
	addConstant(a uintptr, cnst float, N int)

	// a = a * weightA + b * weightB
	linearCombination(a, b uintptr, weightA, weightB float, N int)

	// partial data reduction (operation = add, max, maxabs, ...)
	// input data size = N
	// output = partially reduced data, usually reduced further on CPU. size = blocks
	// patially reduce in "blocks" blocks, partial results in output. blocks = divUp(N, threadsPerBlock*2)
	// use "threads" threads per block: @warning must be < N
	// size "N" of input data, must be > threadsPerBlock
	reduce(operation int, input, output uintptr, buffer *float, blocks, threads, N int) float

	// normalizes a vector field. N = length of one component
	normalize(m uintptr, N int)

	// normalizes a vector field and multiplies with normMap. N = length of one component = length of normMap
	normalizeMap(m, normMap uintptr, N int)

	// overwrites h with torque(m, h) * dtGilbert. N = length of one component
	deltaM(m, h uintptr, alpha, dtGilbert float, N int)

	// Override the GPU stride, handy for debugging. -1 Means reset to the original GPU stride
	// TODO: get rid of? decide the stride by yourself instead of globally storing it?
	overrideStride(nFloats int)

	//____________________________________________________________________ tensor (safe wrappers in tensor.go)

	copyPadded(source, dest uintptr, sourceSize, destSize []int, direction int)

	// Allocates an array of floats on the Device.
	// By convention, Device arrays are represented by an uintptr,
	// while host arrays are *float's.
	// Does not need to be initialized with zeros
	newArray(nFloats int) uintptr

	// Frees device memory allocated by newArray
	freeArray(ptr uintptr)

	// Copies nFloats to, on or from the device, depending on the direction flag (1, 2 or 3)
	memcpy(source, dest uintptr, nFloats, direction int)

	// Offset the array pointer by "index" floats, useful for taking sub-arrays
	// TODO: on a multi-device this will not work
	arrayOffset(array uintptr, index int) uintptr

	// Overwrite n floats with zeros
	zero(data uintptr, nFloats int)

	//____________________________________________________________________ specialized (used in only one place)


	semianalStep(m, h uintptr, dt, alpha float, order, N int)

	// Extract only the real parts form in interleaved complex array
	// 	extractReal(complex, real uintptr, NReal int)

	// Automatically selects between kernelmul3/4/6
	kernelMul(mx, my, mz, kxx, kyy, kzz, kyz, kxz, kxy uintptr, kerntype, nRealNumbers int)

	// In-place kernel multiplication (m gets overwritten by h).
	// The kernel is symmetric so only 6 of the 9 components need to be passed (xx, yy, zz, yz, xz, xy).
	// The kernel is also purely real, so the imaginary parts do not have to be stored (TODO)
	// This is the typical situation for a 3D micromagnetic problem
	// kernelMul6(mx, my, mz, kxx, kyy, kzz, kyz, kxz, kxy uintptr, nRealNumbers int)

	// In-place kernel multiplication (m gets overwritten by h).
	// The kernel is symmetric and contains no mixing between x and (y, z),
	// so only 4 of the 9 components need to be passed (xx, yy, zz, yz).
	// The kernel is also purely real, so the imaginary parts do not have to be stored (TODO)
	// This is the typical situation for a finite 2D micromagnetic problem
	// TODO
	// kernelMul4(mx, my, mz, kxx, kyy, kzz, kyz uintptr, nRealNumbers int)

	// In-place kernel multiplication (m gets overwritten by h).
	// The kernel is symmetric and contains no x contributions.
	// so only 3 of the 9 components need to be passed (yy, zz, yz).
	// The kernel is also purely real, so the imaginary parts do not have to be stored (TODO)
	// This is the typical situation for a infinitely thick 2D micromagnetic problem,
	// which has no demag effects in the out-of-plane direction
	// TODO
	// kernelMul3(my, mz, kyy, kzz, kyz uintptr, nRealNumbers int)

	// unsafe creation of C fftPlan
	newFFTPlan(dataSize, logicSize []int) uintptr

	fft(plan uintptr, in, out uintptr, direction int)

	//______________________________________________________________________________ already safe

	// The GPU stride in number of floats (!)
	Stride() int

	// Bytes allocated on the device
	UsedMem() uint64

	// Print the GPU properties to stdout
	// TODO: return string
	PrintProperties()

	TimerPrintDetail()

	String() string
}

// direction flag for memcpy()
const (
	CPY_TO   = 1
	CPY_ON   = 2
	CPY_FROM = 3
)

// direction flag for copyPadded()
const (
	CPY_PAD   = 1
	CPY_UNPAD = 2
)

// direction flag for FFT
const (
	FFT_FORWARD = 1
	FFT_INVERSE = -1
)

// Reduction operation flags for reduce()
const (
	ADD    = 1
	MAX    = 2
	MAXABS = 3
)
