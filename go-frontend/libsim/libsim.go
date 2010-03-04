package libsim

/*
#include "../../libfft/libfft.h"
//#include "../../libtensor/libtensor.h"

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
import . "fmt"
//import "os"

type real float;

func init(){ // must be lower case!
  Println("initializing libsim...");
  // here we could also set the number of cpu's
  FFTInit();
  Println("...libsim initialized.");
}

func IntToBytes(i int, bytes []byte){
  source := unsafe.Pointer(&i);
  dest := unsafe.Pointer(&(bytes[0]));
  C.array_copy(source, dest, 1);
}

func FloatToBytes(i float, bytes []byte){
  source := unsafe.Pointer(&i);
  dest := unsafe.Pointer(&(bytes[0]));
  C.array_copy(source, dest, 1);
}

/** used to check alignments. */
func ToInt(pointer unsafe.Pointer) int64{
  return (int64)(C.pointer_to_int(pointer));
}

// func WriteInt(i int, out os.File){
//   //C.write_int(_C_int(i), _C_int(out.Fd()));
// }
// 
// func WriteFloat(f float, out os.File){
// 
// }


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

