package gpu

import(
  "unsafe"
  "tensor"
)

const(
  X = 0
  Y = 1
  Z = 2
)

/// a Tensor whose data resides on the GPU, implements tensor.Tensor.
type Tensor struct{
  size []int
  data unsafe.Pointer   // points to float array on the GPU
}

func NewTensor(size []int) *Tensor{
  t := new(Tensor)
  t.size = make([]int, len(size))
  length := 1
  for i:= range size {
    t.size[i] = size[i]
    length *= size[i]
  }
  t.data = NewArray(length)
  return t
}

func (t *Tensor) Size() []int{
  return t.size
}

func (t *Tensor) Get(index []int) float{
  i := tensor.Index(t.size, index)
  return ArrayGet(t.data, i)
}

func Len(size []int) int{
  length := 1
  for i:=range size{
    length *= size[i]
  }
  return length
}

/// copies between two Tensors on the gpu
func TensorCopyOn(source, dest *Tensor){
  assert(tensor.EqualSize(source.size, dest.size))
  MemcpyOn(source.data, dest.data, tensor.Len(source));
}

/// copies a tensor to the GPU
func TensorCopyTo(source tensor.StoredTensor, dest *Tensor){
  ///@todo gpu.Set(), allow tensor.Tensor source, type switch for efficient copying
  ///@todo TensorCpy() with type switch for auto On/To/From
  assert(tensor.EqualSize(source.Size(), dest.size))
  MemcpyTo(&(source.List()[0]), dest.data, tensor.Len(source));
}

/// copies a tensor to the GPU
func TensorCopyFrom(source *Tensor, dest tensor.StoredTensor){
  ///@todo gpu.Set(), allow tensor.Tensor source, type switch for efficient copying
  ///@todo TensorCpy() with type switch for auto On/To/From
  assert(tensor.EqualSize(source.Size(), dest.Size()))
  MemcpyFrom(source.data, &(dest.List()[0]), tensor.Len(source));
}
