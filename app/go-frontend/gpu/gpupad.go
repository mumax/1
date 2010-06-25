package gpu

/*
#include "../../../core/gpupad.h"
*/
import "C"
//import "unsafe"


func CopyPad(source, dest *Tensor){
  C.gpu_copy_pad_unsafe((*_C_float)(source.data), (*_C_float)(dest.data),
                        _C_int(source.size[0]), _C_int(source.size[1]), _C_int(source.size[2]),
                        _C_int(  dest.size[0]), _C_int(  dest.size[1]), _C_int(  dest.size[2]))
}

