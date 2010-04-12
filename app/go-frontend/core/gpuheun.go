package core

/*
#include "../../../core/gpuheun.h"
*/
import "C"
import "unsafe"

import . "../tensor"

type GpuHeun struct{
  pointer *_C_gpuheun
}

func NewGpuHeun(N0, N1, N2 int, kernel StoredTensor, hExt []float) *GpuHeun{
  pointer := C.new_gpuheun((_C_int)(N0), (_C_int)(N1), (_C_int)(N2), ToCTensor(kernel), (*_C_float)(unsafe.Pointer(&hExt[0])));
  return &GpuHeun{pointer};
}

func (solver *GpuHeun) Step(dt float){
  C.gpuheun_step(solver.pointer, (_C_float)(dt));
}
