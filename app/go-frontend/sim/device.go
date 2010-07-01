package sim

import(
  "unsafe"
)

/**
 * The Device interface makes an abstraction from a library with
 * basic simulation functions for a specific computing device
 * like a GPU or CPU (or possibly even a cluster).
 *
 * The interface specifies quite a number of simulation primitives
 * like fft's, torque(), memory allocation... where all higher-level
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
type Device interface{

  torque(m, h unsafe.Pointer, alpha, dtGilbert float, N int)

  normalize(m unsafe.Pointer, N int)

  normalizeMap(m, normMap unsafe.Pointer, N int)

  eulerStage(m, torque unsafe.Pointer, N int)

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


  // Copies from a smaller to a larger tensor, not touching the additional space in the destination (typically filled with zero padding)
  copyPad(source, dest unsafe.Pointer, sourceSize, destSize []int)

  //Copies from a larger to a smaller tensor, not reading the additional data in the source (typically filled with zero padding or spoiled data)
  copyUnpad(source, dest unsafe.Pointer, sourceSize, destSize []int)

  // unsafe creation of C fftPlan
  newFFTPlan(dataSize, logicSize []int) unsafe.Pointer

  // unsafe FFT
  fftForward(plan unsafe.Pointer, in, out unsafe.Pointer)

  // unsafe FFT
  fftInverse(plan unsafe.Pointer, in, out unsafe.Pointer)

  /**
  * Allocates an array of floats on the GPU.
  * By convention, GPU arrays are represented by an unsafe.Pointer,
  * while host arrays are *float's.
  * Initialized with zeros
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

  /// The GPU stride in number of floats (!)
  Stride() int

  /// Override the GPU stride, handy for debugging. -1 Means reset to the original GPU stride
  overrideStride(nFloats int)

  /// Overwrite n floats with zeros
  zero(data unsafe.Pointer, nFloats int)

  /// Print the GPU properties to stdout
  PrintProperties()

}
