package core

/*
#include "../../../core/gpuheun.h"
*/
import "C"
import "unsafe"

import . "../tensor"

type GpuHeun struct{
  size []int;
  pointer *_C_gpuheun
}

func NewGpuHeun(N0, N1, N2 int, kernel StoredTensor, hExt []float) *GpuHeun{
  pointer := C.new_gpuheun((_C_int)(N0), (_C_int)(N1), (_C_int)(N2), ToCTensor(kernel), (*_C_float)(unsafe.Pointer(&hExt[0])));
  return &GpuHeun{[]int{N0, N1, N2}, pointer};
}

func (solver *GpuHeun) Step(dt, alpha float){
  C.gpuheun_step(solver.pointer, (_C_float)(dt), (_C_float)(alpha));
}

func (solver *GpuHeun) LoadM(m StoredTensor){
  assertEqualSize(solver.size, m.Size());
  C.gpuheun_loadm(solver.pointer, ToCTensor(m));
}

func (solver *GpuHeun) StoreM(m StoredTensor){
  assertEqualSize(solver.size, m.Size());
  C.gpuheun_storem(solver.pointer, ToCTensor(m));
}

func (solver *GpuHeun) StoreH(h StoredTensor){
  assertEqualSize(solver.size, h.Size());
  C.gpuheun_storeh(solver.pointer, ToCTensor(h));
}


