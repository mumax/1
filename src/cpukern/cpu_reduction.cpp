#include "cpu_reduction.h"
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

/// @todo parallellize
void cpu_sum(float* input, float* output, int N){
  float sum = 0.0;
  for(int i=0; i<N; i++){
    sum += input[i];
  }
  output[0] = sum;
}

/// unlike the GPU version, we put everything in output[0]
void cpu_reduce(int operation, float* input, float* output, int blocks, int threadsPerBlock, int N){

  for(int i=0; i<blocks; i++){
    output[i] = 0.;
  }
  
  switch(operation){
    default: abort(); break;
    case REDUCE_ADD: cpu_sum(input, output, N);
  }
}


#ifdef __cplusplus
}
#endif
