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

func NewTensor(b Backend, size []int) *Tensor{
  t := new(Tensor)
  
  t.size = make([]int, len(size))
  length := 1
  for i:= range size {
    t.size[i] = size[i]
    length *= size[i]
  }
  t.data = b.newArray(length)
  return t
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


/// copies between two Tensors on the sim
func TensorCopyOn(source, dest *Tensor){
  assert(tensor.EqualSize(source.size, dest.size))
  source.memcpyOn(source.data, dest.data, tensor.N(source));
}

/// copies a tensor to the GPU
func TensorCopyTo(source tensor.StoredTensor, dest *Tensor){
  ///@todo sim.Set(), allow tensor.Tensor source, type switch for efficient copying
  ///@todo TensorCpy() with type switch for auto On/To/From
  assert(tensor.EqualSize(source.Size(), dest.size))
  dest.memcpyTo(&(source.List()[0]), dest.data, tensor.N(source));
}

/// copies a tensor to the GPU
func TensorCopyFrom(source *Tensor, dest tensor.StoredTensor){
  ///@todo sim.Set(), allow tensor.Tensor source, type switch for efficient copying
  ///@todo TensorCpy() with type switch for auto On/To/From
  assert(tensor.EqualSize(source.Size(), dest.Size()))
  source.memcpyFrom(source.data, &(dest.List()[0]), tensor.N(source));
}

func ZeroTensor(t *Tensor){
  t.zero(t.data, tensor.N(t));
}

func CopyPad(source, dest *Tensor){
  source.copyPad(source.data, dest.data, source.size, dest.size)
}

func CopyUnpad(source, dest *Tensor){
  source.copyUnpad(source.data, dest.data, source.size, dest.size)
}

//___________________________________________________________________________________________________ go utilities

// ToCTensor(t tensor.StoredTensor) *_C_tensor{
//   return C.as_tensorN((*_C_float)(unsafe.Pointer(&(t.List()[0]))), (_C_int)(tensor.Rank(t)), (*_C_int)(unsafe.Pointer(&(t.Size()[0]))) );
// }
//
// ToCGPUTensor(t *Tensor) *_C_tensor{
//   return C.as_tensorN((*_C_float)(t.data), (_C_int)(tensor.Rank(t)), (*_C_int)(unsafe.Pointer(&(t.Size()[0]))) );
// }



