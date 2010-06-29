package sim

import(
  "tensor"
)

type Backend struct{
  Device
}


func(b Backend)  NewTensor(size []int) *Tensor{
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


/// copies between two Tensors on the sim
func(b Backend)  TensorCopyOn(source, dest *Tensor){
  assert(tensor.EqualSize(source.size, dest.size))
  b.memcpyOn(source.data, dest.data, tensor.N(source));
}

/// copies a tensor to the GPU
func(b Backend)  TensorCopyTo(source tensor.StoredTensor, dest *Tensor){
  ///@todo sim.Set(), allow tensor.Tensor source, type switch for efficient copying
  ///@todo TensorCpy() with type switch for auto On/To/From
  assert(tensor.EqualSize(source.Size(), dest.size))
  b.memcpyTo(&(source.List()[0]), dest.data, tensor.N(source));
}

/// copies a tensor to the GPU
func(b Backend)  TensorCopyFrom(source *Tensor, dest tensor.StoredTensor){
  ///@todo sim.Set(), allow tensor.Tensor source, type switch for efficient copying
  ///@todo TensorCpy() with type switch for auto On/To/From
  assert(tensor.EqualSize(source.Size(), dest.Size()))
  b.memcpyFrom(source.data, &(dest.List()[0]), tensor.N(source));
}


func(b Backend)  ZeroTensor(t *Tensor){
  b.zero(t.data, tensor.N(t));
}


func (b Backend) CopyPad(source, dest *Tensor){
  b.copyPad(source.data, dest.data, source.size, dest.size)
}

func (b Backend) CopyUnpad(source, dest *Tensor){
  b.copyUnpad(source.data, dest.data, source.size, dest.size)
}

// Takes an array size and returns the smallest multiple of Stride() where the array size fits in
func(b Backend)  PadToStride(nFloats int) int{
  stride := b.stride()
  gpulen := ((nFloats-1)/stride + 1) * stride;

  assert(gpulen % stride == 0)
  assert(gpulen > 0)
  assert(gpulen >= nFloats)
  return gpulen
}
