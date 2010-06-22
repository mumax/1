package gpu

/*
#include "../../../core/tensor.h"
*/
import "C"
import "unsafe"

import(
  "tensor"
  "log"
)

func ToCTensor(t tensor.StoredTensor) *_C_tensor{
  return C.as_tensorN((*_C_float)(unsafe.Pointer(&(t.List()[0]))), (_C_int)(tensor.Rank(t)), (*_C_int)(unsafe.Pointer(&(t.Size()[0]))) );
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
