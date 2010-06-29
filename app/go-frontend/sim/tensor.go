package sim

import(
  "unsafe"
  "tensor"
)

const(
  X = 0
  Y = 1
  Z = 2
)


type Tensor struct{
  Backend               ///< wraps the Device where the Tensor resides on (GPU/CPU/...)
  size []int
  data unsafe.Pointer   // points to float array on the GPU/CPU
}


func(t *Tensor) Get(index []int) float{
  i := tensor.Index(t.size, index)
  return t.arrayGet(t.data, i)
}


func(t *Tensor) Set(index []int, value float){
  i := tensor.Index(t.size, index)
  t.arraySet(t.data, i, value)
}

func(t *Tensor) Size() []int{
  return t.size
}


func Len(size []int) int{
  length := 1
  for i:=range size{
    length *= size[i]
  }
  return length
}


func assertEqualSize(sizeA, sizeB []int){
  assert(len(sizeA) == len(sizeB));
  for i:=range(sizeA){
    assert(sizeA[i] == sizeB[i]);
  }
}

//___________________________________________________________________________________________________ go utilities

// ToCTensor(t tensor.StoredTensor) *_C_tensor{
//   return C.as_tensorN((*_C_float)(unsafe.Pointer(&(t.List()[0]))), (_C_int)(tensor.Rank(t)), (*_C_int)(unsafe.Pointer(&(t.Size()[0]))) );
// }
//
// ToCGPUTensor(t *Tensor) *_C_tensor{
//   return C.as_tensorN((*_C_float)(t.data), (_C_int)(tensor.Rank(t)), (*_C_int)(unsafe.Pointer(&(t.Size()[0]))) );
// }



