package sim

import (
	"unsafe"
)

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

	//____________________________________________________________________ general purpose (use Backend safe wrappers)

	// adds b to a. N = length of a = length of b
	add(a, b unsafe.Pointer, N int)

	// adds the constant cnst to a. N = length of a
	addConstant(a unsafe.Pointer, cnst float, N int)

	// a = a * weightA + b * weightB
	linearCombination(a, b unsafe.Pointer, weightA, weightB float, N int)

	// normalizes a vector field. N = length of one component
	normalize(m unsafe.Pointer, N int)

	// normalizes a vector field and multiplies with normMap. N = length of one component = length of normMap
	normalizeMap(m, normMap unsafe.Pointer, N int)

	// overwrites h with torque(m, h) * dtGilbert. N = length of one component
	deltaM(m, h unsafe.Pointer, alpha, dtGilbert float, N int)

	/// Override the GPU stride, handy for debugging. -1 Means reset to the original GPU stride
	overrideStride(nFloats int)

	//____________________________________________________________________ tensor (safe wrappers in tensor.go)

	// Copies from a smaller to a larger tensor, not touching the additional space in the destination (typically filled with zero padding)
	copyPad(source, dest unsafe.Pointer, sourceSize, destSize []int)

	//Copies from a larger to a smaller tensor, not reading the additional data in the source (typically filled with zero padding or spoiled data)
	copyUnpad(source, dest unsafe.Pointer, sourceSize, destSize []int)

	/**
	 * Allocates an array of floats on the GPU.
	 * By convention, GPU arrays are represented by an unsafe.Pointer,
	 * while host arrays are *float's.
	 * Does not need to be initialized with zeros
	 */
	newArray(nFloats int) unsafe.Pointer

	/// Copies a number of floats from host to GPU
	memcpyTo(source *float, dest unsafe.Pointer, nFloats int)

	/// Copies a number of floats from GPU to host
	memcpyFrom(source unsafe.Pointer, dest *float, nFloats int)

	/// Copies a number of floats from GPU to GPU
	memcpyOn(source, dest unsafe.Pointer, nFloats int)

	/// Gets one float from a GPU array
	arrayGet(array unsafe.Pointer, index int) float

	arraySet(array unsafe.Pointer, index int, value float)

	arrayOffset(array unsafe.Pointer, index int) unsafe.Pointer

	/// Overwrite n floats with zeros
	zero(data unsafe.Pointer, nFloats int)

	//____________________________________________________________________ specialized (used in only one place)


	semianalStep(m, h unsafe.Pointer, dt, alpha float, N int)

	// In-place kernel multiplication (m gets overwritten by h).
	// The kernel is symmetric so only 6 of the 9 components need to be passed (xx, yy, zz, yz, xz, xy).
	// The kernel is also purely real, so the imaginary parts do not have to be stored (TODO)
	// This is the typical situation for a 3D micromagnetic problem
	kernelMul6(mx, my, mz, kxx, kyy, kzz, kyz, kxz, kxy unsafe.Pointer, nRealNumbers int)

	// In-place kernel multiplication (m gets overwritten by h).
	// The kernel is symmetric and contains no mixing between x and (y, z),
	// so only 4 of the 9 components need to be passed (xx, yy, zz, yz).
	// The kernel is also purely real, so the imaginary parts do not have to be stored (TODO)
	// This is the typical situation for a finite 2D micromagnetic problem
	// TODO
	// kernelMul4(mx, my, mz, kxx, kyy, kzz, kyz unsafe.Pointer, nRealNumbers int)

	// In-place kernel multiplication (m gets overwritten by h).
	// The kernel is symmetric and contains no x contributions.
	// so only 3 of the 9 components need to be passed (yy, zz, yz).
	// The kernel is also purely real, so the imaginary parts do not have to be stored (TODO)
	// This is the typical situation for a infinitely thick 2D micromagnetic problem,
	// which has no demag effects in the out-of-plane direction
	// TODO
	// kernelMul3(my, mz, kyy, kzz, kyz unsafe.Pointer, nRealNumbers int)

	// unsafe creation of C fftPlan
	newFFTPlan(dataSize, logicSize []int) unsafe.Pointer

	// unsafe FFT
	fftForward(plan unsafe.Pointer, in, out unsafe.Pointer)

	// unsafe FFT
	fftInverse(plan unsafe.Pointer, in, out unsafe.Pointer)

	//______________________________________________________________________________ already safe

	/// The GPU stride in number of floats (!)
	Stride() int

	/// Print the GPU properties to stdout
	PrintProperties()

	String() string
}
