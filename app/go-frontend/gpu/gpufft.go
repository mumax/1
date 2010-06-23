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
  plan unsafe.Pointer
  logicSize [3]int
  dataSize [3]int
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
    fft.logicSize[i] = logicSize[i]
    fft.dataSize [i] = dataSize[i]
  }
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
  return PhysicSize(fft.logicSize[0:]);
}





func PhysicSize(logicSize []int) []int{
  assert(len(logicSize) == 3)
  physicSize := make([]int, 3)
  physicSize[X] = logicSize[X]
  physicSize[Y] = logicSize[Y]
  physicSize[Z] = PadToStride(logicSize[Z] + 2)
  return physicSize;
}
