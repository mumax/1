package gpu

/*
#include "../../../core/gpufft2.h"
*/
import "C"
import "unsafe"


type FFT struct{
  plan unsafe.Pointer
}

func NewFFT(logicSize []int) *FFT{
  assert(len(logicSize) == 3)
  return &FFT{ unsafe.Pointer( C.new_gpuFFT3dPlan((*_C_int)(unsafe.Pointer(&logicSize[0]))) ) }
}

func NewFFTPadded(logicSize, logicDataSize []int) *FFT{
  assert(len(logicSize) == 3)
  assert(len(logicDataSize) == 3)
  for i:=range logicSize{ assert(logicSize[i] >= logicDataSize[i]) }
  return &FFT{ unsafe.Pointer( C.new_gpuFFT3dPlan((*_C_int)(unsafe.Pointer(&logicSize[0]))) ) }
}

