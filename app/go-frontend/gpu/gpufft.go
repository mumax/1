package gpu

/*
#include "../../../core/gpufft2.h"
*/
import "C"
import "unsafe"

import(
//   "tensor"
)

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


