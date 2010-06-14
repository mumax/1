package core

/*
#include "../../../core/gpuheun2.h"
*/
import "C"
import "unsafe"

import . "../tensor"

type GpuHeun struct{
  size []int;
  pointer *_C_gpuheun2
}

func NewGpuHeun(size []int, kernel StoredTensor, hExt []float) *GpuHeun{
  pointer := C.new_gpuheun2((*_C_int)(unsafe.Pointer(&size[0])), ToCTensor(kernel), (*_C_float)(unsafe.Pointer(&hExt[0])));
  return &GpuHeun{size, pointer};       ///@todo a copy of size might be safer
}

func (solver *GpuHeun) Step(dt, alpha float){
  C.gpuheun2_step(solver.pointer, (_C_float)(dt), (_C_float)(alpha));
}

func (solver *GpuHeun) LoadM(m StoredTensor){
  assertEqualSize(solver.size, []int{m.Size()[1+X], m.Size()[1+Y], m.Size()[1+Z]});
  C.gpuheun2_loadm(solver.pointer, ToCTensor(m));
}

func (solver *GpuHeun) StoreM(m StoredTensor){
  assertEqualSize(solver.size, []int{m.Size()[1+X], m.Size()[1+Y], m.Size()[1+Z]});
  C.gpuheun2_storem(solver.pointer, ToCTensor(m));
}

func (solver *GpuHeun) StoreH(h StoredTensor){
  assertEqualSize(solver.size, []int{h.Size()[1+X], h.Size()[1+Y], h.Size()[1+Z]});
  C.gpuheun2_storeh(solver.pointer, ToCTensor(h));
}


