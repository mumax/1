#include "gputil.h"
#include "timer.h"
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <stdarg.h>

#ifdef __cplusplus
extern "C" {
#endif

//_____________________________________________________________________________________________ data management


tensor* new_gputensor(int rank, int* size){
  int len = 1;
  for(int i=0; i<rank; i++){
    len *= size[i];
  }
  return as_tensorN(new_gpu_array(len), rank, size);
}

void memcpy_gpu_to_gpu(float* source, float* dest, int nElements){
  
  assert(nElements > 0);
  int status = cudaMemcpy(dest, source, nElements*sizeof(float), cudaMemcpyDeviceToDevice);
  if(status != cudaSuccess){
    fprintf(stderr, "CUDA could not copy %d floats from device addres %p to device addres %p\n", nElements, source, dest);
    gpu_safe(status);
  }
  gpu_sync();
  
}

void tensor_copy_to_gpu(tensor* source, tensor* dest){
  assert(tensor_equalsize(source, dest));
  memcpy_to_gpu(source->list, dest->list, source->len);
}

void tensor_copy_from_gpu(tensor* source, tensor* dest){
  assert(tensor_equalsize(source, dest));
  memcpy_from_gpu(source->list, dest->list, source->len);
}

void tensor_copy_gpu_to_gpu(tensor* source, tensor* dest){
  assert(tensor_equalsize(source, dest));
  memcpy_gpu_to_gpu(source->list, dest->list, source->len);
}



#ifdef __cplusplus
}
#endif