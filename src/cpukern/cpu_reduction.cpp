#include "cpu_reduction.h"
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

/// @todo parallellize
float cpu_sum(float* input, int N){
  float sum = 0.0;
  for(int i=0; i<N; i++){
    sum += input[i];
  }
  return sum;
}


/// Reduces the input (array on device)
float cpu_reduce(int operation, float* input, float* devbuffer, float* hostbuffer, int blocks, int threadsPerBlock, int N){
  switch(operation){
    default: abort();
    case REDUCE_ADD: return cpu_sum(input, N);
  }
}


#ifdef __cplusplus
}
#endif
