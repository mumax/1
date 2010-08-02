#include "cpu_linalg.h"

#ifdef __cplusplus
extern "C" {
#endif


void cpu_add(float* a, float* b, int N){
  for(int i=0; i<N; i++){
    a[i] += b[i];
  }
}


void cpu_add_constant(float* a, float cnst, int N){
  for(int i=0; i<N; i++){
    a[i] += cnst;
  }
}


void cpu_linear_combination(float* a, float* b, float weightA, float weightB, int N){ 
  for(int i=0; i<N; i++){
    a[i] = weightA * a[i] + weightB * b[i];
  }
}

#ifdef __cplusplus
}
#endif
