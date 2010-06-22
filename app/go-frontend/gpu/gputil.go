package gpu

/*
#include "../../../core/gputil.h"

*/
import "C"
import "unsafe"

func NewArray(nFloats int) unsafe.Pointer{
  return unsafe.Pointer(C.new_gpu_array(_C_int(nFloats)))
}

func PrintProperties(){
  C.print_device_properties((*_C_FILE)(C.stderr));
}




