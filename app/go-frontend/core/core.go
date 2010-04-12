package core

/*
#include "../../../core/tensor.h"
#include "../../../core/gpuheun.h"

*/
import "C"
import "unsafe"

import . "../tensor"

func ToCTensor(t StoredTensor) *_C_tensor{
  return C.as_tensorN((*_C_float)(unsafe.Pointer(&(t.List()[0]))), (_C_int)(Rank(t)), (*_C_int)(unsafe.Pointer(&(t.Size()[0]))) );
}
