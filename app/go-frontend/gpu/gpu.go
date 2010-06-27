package gpu

/*
#include "../../../core/tensor.h"
#include "../../../core/gputil.h"
#include "../../../core/gpufft2.h"
#include "../../../core/gpupad.h"
#include "../../../core/gpuconv2.h"
*/
import "C"
import "unsafe"

/**
 * This single file intefaces all the relevant CUDA functions with go
 * 
 * @note cgo does not seem to like many cgofiles, so I put everything together here.
 * @author Arne Vansteenkiste
 */

import(
  "tensor"
  "log"
)

//___________________________________________________________________________________________________ Kernel multiplication

//
// from gpuconv2.h:
// void gpu_kernel_mul_complex_inplace_symm(float* fftMx,  float* fftMy,  float* fftMz,
//                                          float* fftKxx, float* fftKyy, float* fftKzz,
//                                          float* fftKyz, float* fftKxz, float* fftKxy,
//                                          int nRealNumbers);

func KernelMul(mx, my, mz, kxx, kyy, kzz, kyz, kxz, kxy unsafe.Pointer, nRealNumbers int){
  C.gpu_kernel_mul_complex_inplace_symm(
        (*_C_float)(mx), (*_C_float)(my), (*_C_float)(mz),
        (*_C_float)(kxx), (*_C_float)(kyy), (*_C_float)(kzz),
        (*_C_float)(kyz), (*_C_float)(kxz), (*_C_float)(kxy),
        _C_int(nRealNumbers))
}

//___________________________________________________________________________________________________ Copy-pad


///Copies from a smaller to a larger tensor, not touching the additional space in the destination (typically filled with zero padding)
func CopyPad(source, dest *Tensor){
  C.gpu_copy_pad_unsafe((*_C_float)(source.data), (*_C_float)(dest.data),
                        _C_int(source.size[0]), _C_int(source.size[1]), _C_int(source.size[2]),
                        _C_int(  dest.size[0]), _C_int(  dest.size[1]), _C_int(  dest.size[2]))
}


///Copies from a larger to a smaller tensor, not reading the additional data in the source (typically filled with zero padding or spoiled data)
func CopyUnpad(source, dest *Tensor){
  C.gpu_copy_unpad_unsafe((*_C_float)(source.data), (*_C_float)(dest.data),
                        _C_int(source.size[0]), _C_int(source.size[1]), _C_int(source.size[2]),
                        _C_int(  dest.size[0]), _C_int(  dest.size[1]), _C_int(  dest.size[2]))
}

//___________________________________________________________________________________________________ FFT

/// 3D real-to-complex / complex-to-real transform. Handles zero-padding efficiently (if applicable)
type FFT struct{
  plan unsafe.Pointer           ///< points to the gpuFFT3dPlan struct that does the actual FFT
  dataSize   [3]int             ///< size of the non-zero data inside the logic input data. Must be <= logicSize
  logicSize  [3]int             ///< logical size of the FFT, including padding: number of reals in each dimension
  physicSize [3]int             ///< The input data needs to be padded with zero's to physicSize, in order to accomodate for the extra complex number in the last dimension needed by real-to-complex FFTS. Additionally, even extra zero's are probably going to be added to fit the gpu stride.
}

/// logicSize is the size of the real input data.
func NewFFT(logicSize []int) *FFT{
  return NewFFTPadded(logicSize, logicSize)
}

/**
 * logicSize is the size of the real input data, but this may contain a lot of zeros.
 * dataSize is the portion of logicSize that is non-zero (typically half as large as logicSize).
 */
func NewFFTPadded(dataSize, logicSize []int) *FFT{
  assert(len(logicSize) == 3)
  assert(len(dataSize) == 3)
  for i:=range dataSize{
    assert(dataSize[i] <= logicSize[i])
  }

  fft := new(FFT)

  for i:=range logicSize {
    fft.logicSize [i] = logicSize[i]
    fft.dataSize  [i] = dataSize[i]
    fft.physicSize[i] = fft.logicSize[i] // Z will be overwritten
  }
  fft.physicSize[Z] = PadToStride(logicSize[Z] + 2)

  Csize := (*_C_int)(unsafe.Pointer(&fft.dataSize[0]))
  CpaddedSize := (*_C_int)(unsafe.Pointer(&fft.logicSize[0]))
  fft.plan = unsafe.Pointer( C.new_gpuFFT3dPlan_padded( Csize, CpaddedSize ) )

  return fft
}


func (fft *FFT) Forward(in, out *Tensor){
  // size checks
  assert(tensor.Rank(in) == 3)
  assert(tensor.Rank(out) == 3)
  for i,s := range fft.physicSize{
    assert(  in.size[i] == s)
    assert( out.size[i] == s)
  }
  // actual fft
  C.gpuFFT3dPlan_forward_unsafe((*_C_gpuFFT3dPlan)(fft.plan), (*_C_float)(in.data), (*_C_float)(out.data))
}

func (fft *FFT) Inverse(in, out *Tensor){
  // size checks
  assert(tensor.Rank(in) == 3)
  assert(tensor.Rank(out) == 3)
  for i,s := range fft.physicSize{
    assert(  in.size[i] == s)
    assert( out.size[i] == s)
  }
  // actual fft
  C.gpuFFT3dPlan_inverse_unsafe((*_C_gpuFFT3dPlan)(fft.plan), (*_C_float)(in.data), (*_C_float)(out.data));
}

func (fft *FFT) Normalization() int{
  return int(C.gpuFFT3dPlan_normalization((*_C_gpuFFT3dPlan)(fft.plan)))
}


//_______________________________________________________________________________ GPU memory allocation

/**
 * Allocates an array of floats on the GPU.
 * By convention, GPU arrays are represented by an unsafe.Pointer,
 * while host arrays are *float's.
 */
func NewArray(nFloats int) unsafe.Pointer{
  return unsafe.Pointer(C.new_gpu_array(_C_int(nFloats)))
}

/// Copies a number of floats from host to GPU
func MemcpyTo(source *float, dest unsafe.Pointer, nFloats int){
  C.memcpy_to_gpu((*_C_float)(unsafe.Pointer(source)), (*_C_float)(dest), _C_int(nFloats));
}

/// Copies a number of floats from GPU to host
func MemcpyFrom(source unsafe.Pointer, dest *float, nFloats int){
  C.memcpy_from_gpu((*_C_float)(source), (*_C_float)(unsafe.Pointer(dest)), _C_int(nFloats));
}

/// Copies a number of floats from GPU to GPU
func MemcpyOn(source, dest unsafe.Pointer, nFloats int){
  C.memcpy_gpu_to_gpu((*_C_float)(source), (*_C_float)(dest), _C_int(nFloats));
}

/// Gets one float from a GPU array
func ArrayGet(array unsafe.Pointer, index int) float{
  return float(C.gpu_array_get((*_C_float)(array), _C_int(index)));
}

func ArraySet(array unsafe.Pointer, index int, value float){
  C.gpu_array_set((*_C_float)(array), _C_int(index), _C_float(value))
}


//___________________________________________________________________________________________________ GPU Stride

/// The GPU stride in number of floats (!)
func Stride() int{
  return int(C.gpu_stride_float());
}

/// Takes an array size and returns the smallest multiple of Stride() where the array size fits in
func PadToStride(nFloats int) int{
  return int(C.gpu_pad_to_stride(_C_int(nFloats)));
}

/// Override the GPU stride, handy for debugging. -1 Means reset to the original GPU stride
func OverrideStride(nFloats int){
  C.gpu_override_stride(_C_int(nFloats));
}

//___________________________________________________________________________________________________ tensor utilities

/// Overwrite n floats with zeros
func Zero(data unsafe.Pointer, nFloats int){
  C.gpu_zero((*_C_float)(data), _C_int(nFloats));
}

func ZeroTensor(t *Tensor){
  Zero(t.data, tensor.Len(t));
}

/// Print the GPU properties to stdout
func PrintProperties(){
  C.print_device_properties_stdout();
}


//___________________________________________________________________________________________________ go utilities

func ToCTensor(t tensor.StoredTensor) *_C_tensor{
  return C.as_tensorN((*_C_float)(unsafe.Pointer(&(t.List()[0]))), (_C_int)(tensor.Rank(t)), (*_C_int)(unsafe.Pointer(&(t.Size()[0]))) );
}

func ToCGPUTensor(t *Tensor) *_C_tensor{
  return C.as_tensorN((*_C_float)(t.data), (_C_int)(tensor.Rank(t)), (*_C_int)(unsafe.Pointer(&(t.Size()[0]))) );
}

func assert(b bool){
  if !b{
    log.Crash("assertion failed");
  }
}

func assertEqualSize(sizeA, sizeB []int){
  assert(len(sizeA) == len(sizeB));
  for i:=range(sizeA){
    assert(sizeA[i] == sizeB[i]);
  }
}
