#include "cpu_normalize.h"
#include <math.h>
#ifdef __cplusplus
extern "C" {
#endif


void cpu_normalize_uniform(float* m, int N){

  float* mx = &(m[0*N]);
  float* my = &(m[1*N]);
  float* mz = &(m[2*N]);

  #pragma omp parallel for
  for(int i=0; i<N; i++){
    float norm = 1.0/sqrt(mx[i]*mx[i] + my[i]*my[i] + mz[i]*mz[i]);
    mx[i] *= norm;
    my[i] *= norm;
    mz[i] *= norm;
  }
  
}


void cpu_normalize_map(float* m, float* map, int N){
  
  float* mx = &(m[0*N]);
  float* my = &(m[1*N]);
  float* mz = &(m[2*N]);

  #pragma omp parallel for
  for(int i=0; i<N; i++){
    float norm = map[i]/sqrt(mx[i]*mx[i] + my[i]*my[i] + mz[i]*mz[i]);
    mx[i] *= norm;
    my[i] *= norm;
    mz[i] *= norm;
  }
  
}

#ifdef __cplusplus
}
#endif
