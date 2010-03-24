package core

/*
#include "../../../core/conv_gpu.h"

// Although this is not libsim code, we put just this one function here. It should be replacable by pure go with some dirty poiter->string->int conversion or so.
// 32-bit vs/ 64-bit issue here:
// 32-bit needs int, 64 bit needs long long int.

int pointer_to_int(void* ptr){
#ifdef _64_BIT
  return (long long int) ptr;
#else
  return (long int) ptr;
#endif
}

void array_copy(void* source, void* dest, int length){
 int i;
 float* s = (float*) source;	// it works for all 32-bit data types
 float* d = (float*) dest;
 for(i=0; i<length; i++){
  d[i] = s[i];
 }
}

*/
import "C"
import "unsafe"

import . "../tensor"
//import . "fmt"
//import . "os"

func init(){ 	// must be lower case!

}

type ConvPlan struct{
  plan unsafe.Pointer
}

func NewConvPlan(size []int, kernel *Tensor5) *ConvPlan{
  return &ConvPlan{new_convplan(size[0], size[1], size[2], DataAddress(kernel))};
}

func ExecuteConv(plan *ConvPlan, source, dest *Tensor4){
  conv_execute(plan.plan, DataAddress(source), DataAddress(dest));
}

func new_convplan(N0, N1, N2 int, kernel unsafe.Pointer) unsafe.Pointer{
  return unsafe.Pointer(C.new_convplan(_C_int(N0), _C_int(N1), _C_int(N2), (*_C_float)(unsafe.Pointer(kernel))));
}

func conv_execute(plan unsafe.Pointer, source, dest unsafe.Pointer){
  C.conv_execute((*_C_convplan)(plan), (*_C_float)(unsafe.Pointer(source)), (*_C_float)(unsafe.Pointer(dest)));
}

/*

func FFTInitForward(N0, N1, N2 int, source, dest unsafe.Pointer) unsafe.Pointer{
  return C.fft_init_forward(_C_int(N0), _C_int(N1), _C_int(N2), (*_C_real)(source), (*_C_real)(dest));
}

func FFTInitBackward(N0, N1, N2 int, source, dest unsafe.Pointer) unsafe.Pointer{
  return C.fft_init_backward(_C_int(N0), _C_int(N1), _C_int(N2), (*_C_real)(source), (*_C_real)(dest));
}

func FFTExecute(plan unsafe.Pointer){
  C.fft_execute(plan);
}

func FFTDestroyPlan(plan unsafe.Pointer){
  C.fft_destroy_plan(plan);
}


func FFTVersion() int{
  return int(C.fft_version());
}*/
