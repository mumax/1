#include "cpu_mem.h"
#include "../macros.h"

#ifdef __cplusplus
extern "C" {
#endif

///@todo use fftwf_malloc
float* new_cpu_array(int size){
  assert(size > 0);
  float* array = (float*)malloc(size * sizeof(float));
  if(array == NULL){
    fprintf(stderr, "could not allocate %d floats in main memory\n", size);
    abort();
  }
  return array;
}

///@todo use fftwf_free
void free_cpu_array(float* array){
  free(array);
}


void cpu_zero(float* data, int nElements){
  for(int i=0; i<nElements; i++){
    data[i] = 0;
  }
}

void cpu_memcpy(float* source, float* dest, int nElements){
  assert(nElements > 0);
  for(int i=0; i<nElements; i++){
    dest[i] = source[i];
  }
}


float cpu_array_get(float* dataptr, int index){
  return dataptr[index];
}


void cpu_array_set(float* dataptr, int index, float value){
  dataptr[index] = value;
}

///@internal
int _cpu_stride_float_cache = -1;

/// We use quadword alignment by default, but allow to override just like on the GPU
int cpu_stride_float(){
  if( _cpu_stride_float_cache == -1){
    _cpu_stride_float_cache = 1; /// The default for now. @todo: this should become 4 when strided FFT's work
  }
  return _cpu_stride_float_cache;
}


void cpu_override_stride(int nFloats){
  assert(nFloats > -2);
  debugv( fprintf(stderr, "CPU stride overridden to %d floats\n", nFloats) );
  _cpu_stride_float_cache = nFloats;
}

// int cpu_pad_to_stride(int nFloats){
//   assert(nFloats > 0);
//   int stride = cpu_stride_float();
//   int cpulen = ((nFloats-1)/stride + 1) * stride;
//   
//   assert(cpulen % stride == 0);
//   assert(cpulen > 0);
//   assert(cpulen >= nFloats);
//   return cpulen;
// }

#ifdef __cplusplus
}
#endif
