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
//import . "fmt"
//import . "os"

type real float;

func init(){ // must be lower case!
//   Fprintln(Stderr, "initializing libsim...");
//   // here we could also set the number of cpu's
//   FFTInit();
//   Fprintln(Stderr, "...libsim initialized.");
}

/** Converts an int to a slice of 4 bytes, using the machine's endianess. */
func IntToBytes(i int, bytes []byte){
  i32 := int32(i); //just to be sure...
  source := unsafe.Pointer(&i32);
  dest := unsafe.Pointer(&(bytes[0]));
  C.array_copy(source, dest, 1);
}

/** Converts a slice of 4 bytes to an int, using the machine's endianess. */
func BytesToInt(bytes []byte) int{
  // idea: check len(bytes) 
  var i int;
  dest := unsafe.Pointer(&i);
  source := unsafe.Pointer(&(bytes[0]));
  C.array_copy(source, dest, 1);
  return i;
}

/** Converts a float to a slice of 4 bytes, using the machine's endianess. */
func FloatToBytes(i float, bytes []byte){
  source := unsafe.Pointer(&i);
  dest := unsafe.Pointer(&(bytes[0]));
  C.array_copy(source, dest, 1);
}

/** Converts a slice of 4 bytes to a float, using the machine's endianess. */
func BytesToFloat(bytes []byte) float{
  var i float;
  dest := unsafe.Pointer(&i);
  source := unsafe.Pointer(&(bytes[0]));
  C.array_copy(source, dest, 1);
  return i;
}

/** Used internally to check alignments. */
func ToInt(pointer unsafe.Pointer) int64{
  return (int64)(C.pointer_to_int(pointer));
}

/*
 * Wrappers for libfft, see libfft.h
 */

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
