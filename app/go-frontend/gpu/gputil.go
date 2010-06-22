package gpu

/*
#include "../../../core/gputil.h"

*/
import "C"
import "unsafe"

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


func Stride() int{
  return int(C.gpu_stride_float());
}

func PadToStride(nFloats int) int{
  return int(C.gpu_pad_to_stride(_C_int(nFloats)));
}

func OverrideStride(nFloats int){
  C.gpu_override_stride(_C_int(nFloats));
}

//_______________________________________________________________________________ util

func Zero(data unsafe.Pointer, nFloats int){
  C.gpu_zero((*_C_float)(data), _C_int(nFloats));
}

func PrintProperties(){
  C.print_device_properties_stdout();
}




