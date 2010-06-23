package gpu

/*
#include "../../../core/gputil.h"
*/
import "C"
import "unsafe"

import(
  "tensor"
)

//_______________________________________________________________________________ memory

/**
 * Allocates an array of floats on the GPU.
 * By convention, GPU arrays are represented by an unsafe.Pointer,
 * while host arrays are *float's.
 */
func NewArray(nFloats int) unsafe.Pointer{
  return unsafe.Pointer(C.new_gpu_array(_C_int(nFloats)))
}

/// Copies a number of floats from host to GPU
func MemcpyTo(source *float, dest unsafe.Pointer, nFloats int){
  C.memcpy_to_gpu((*_C_float)(unsafe.Pointer(source)), (*_C_float)(dest), _C_int(nFloats));
}

/// Copies a number of floats from GPU to host
func MemcpyFrom(source unsafe.Pointer, dest *float, nFloats int){
  C.memcpy_from_gpu((*_C_float)(source), (*_C_float)(unsafe.Pointer(dest)), _C_int(nFloats));
}

/// Copies a number of floats from GPU to GPU
func MemcpyOn(source, dest unsafe.Pointer, nFloats int){
  C.memcpy_gpu_to_gpu((*_C_float)(source), (*_C_float)(dest), _C_int(nFloats));
}

/// Gets one float from a GPU array
func Get(array unsafe.Pointer, index int) float{
  return float(C.gpu_get((*_C_float)(array), _C_int(index)));
}

//_______________________________________________________________________________ tensor

const(
  X = 0
  Y = 1
  Z = 2
)

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
  return Get(t.data, i)
}

//_______________________________________________________________________________ stride

/// The GPU stride in number of floats (!)
func Stride() int{
  return int(C.gpu_stride_float());
}

/// Takes an array size and returns the smallest multiple of Stride() where the array size fits in
func PadToStride(nFloats int) int{
  return int(C.gpu_pad_to_stride(_C_int(nFloats)));
}

/// Override the GPU stride, handy for debugging. -1 Means reset to the original GPU stride
func OverrideStride(nFloats int){
  C.gpu_override_stride(_C_int(nFloats));
}

//_______________________________________________________________________________ util

/// Overwrite n floats with zeros
func Zero(data unsafe.Pointer, nFloats int){
  C.gpu_zero((*_C_float)(data), _C_int(nFloats));
}

/// Print the GPU properties to stdout
func PrintProperties(){
  C.print_device_properties_stdout();
}




