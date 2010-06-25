package gpu

/*
#include "../../../core/gpufft2.h"
#include "../../../core/gpupad.h"
#include "../../../core/gpuconv2.h"
*/
import "C"
import "unsafe"

import(
//   "tensor"
)

// from gpuconv2.h
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

///@todo belongs in gpupad.go, but does not compile there
///Copies from a smaller to a larger tensor, not touching the additional space in the destination (typically filled with zero padding)
func CopyPad(source, dest *Tensor){
  C.gpu_copy_pad_unsafe((*_C_float)(source.data), (*_C_float)(dest.data),
                        _C_int(source.size[0]), _C_int(source.size[1]), _C_int(source.size[2]),
                        _C_int(  dest.size[0]), _C_int(  dest.size[1]), _C_int(  dest.size[2]))
}

///@todo belongs in gpupad.go, but does not compile there
///Copies from a larger to a smaller tensor, not reading the additional data in the source (typically filled with zero padding or spoiled data)
func CopyUnpad(source, dest *Tensor){
  C.gpu_copy_unpad_unsafe((*_C_float)(source.data), (*_C_float)(dest.data),
                        _C_int(source.size[0]), _C_int(source.size[1]), _C_int(source.size[2]),
                        _C_int(  dest.size[0]), _C_int(  dest.size[1]), _C_int(  dest.size[2]))
}


/// 3D real-to-complex / complex-to-real transform. Handles zero-padding efficiently (if applicable)
type FFT struct{
  plan unsafe.Pointer           ///< points to the gpuFFT3dPlan struct that does the actual FFT
  logicSize  [3]int             ///< logical size of the FFT, including padding: number of reals in each dimension
  dataSize   [3]int             ///< size of the non-zero data inside the logic input data. Must be <= logicSize
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
func NewFFTPadded(logicSize, dataSize []int) *FFT{
  assert(len(logicSize) == 3)
  assert(len(dataSize) == 3)
  
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

/**
 * Returns the physical size (needed for storage) corresponding to this
 * FFT's logical size. It is at least 2 floats larger in the Z dimension,
 * and probably even more due to GPU striding.
 */
func (fft *FFT) PhysicSize() []int{
  return fft.physicSize[0:]
}


func (fft *FFT) Forward(in, out *Tensor){
  
  C.gpuFFT3dPlan_forward_unsafe((*_C_gpuFFT3dPlan)(fft.plan), (*_C_float)(in.data), (*_C_float)(out.data))
}

func (fft *FFT) Inverse(in, out *Tensor){
  
  C.gpuFFT3dPlan_inverse_unsafe((*_C_gpuFFT3dPlan)(fft.plan), (*_C_float)(in.data), (*_C_float)(out.data));
}

func (fft *FFT) Normalization() int{
  return int(C.gpuFFT3dPlan_normalization((*_C_gpuFFT3dPlan)(fft.plan)))
}


