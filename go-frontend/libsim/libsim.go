package libsim

/*
#include "../../libfft/libfft.h"

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
*/
import "C"
import "unsafe"
import . "fmt"

type real float;

func Init(){ // does not seem to be called automatically
  Println("initializing libsim...");
  FFTInit();
  Println("...libsim initialized.");
}

/** used to check alignments. */
func ToInt(pointer unsafe.Pointer) int64{
  return (int64)(C.pointer_to_int(pointer));
}

func FFTInit(){
  C.fft_init();
}

func FFTFinalize(){
  C.fft_finalize();
}

/** aligned malloc, not used by go but here to have libsim wrapped completely. */ 
func FFTMalloc(N0, N1, N2 int) unsafe.Pointer{
  return unsafe.Pointer( C.fft_malloc(_C_int(N0), _C_int(N1), _C_int(N2)));
}

/** free memory allocated with FFTMalloc, not necessary for go. */ 
func FFTFree(data unsafe.Pointer){
  C.fft_free(data);
}

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
}

