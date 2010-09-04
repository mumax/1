#include "gpukern.h"
#include "timer.h"
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>




int main(){
  int N0=2, N1=4, N2=8;
  int N = N0*N1*N2;
  int* size = new int[3];
  size[0] = N0;
  size[1] = N1;
  size[2] = N2;

  gpuFFT3dPlan* plan = new_gpuFFT3dPlan_padded(size, size);
    
  float* host1 = (float*)calloc(N, sizeof(float));
  float* host2 = (float*)calloc(N, sizeof(float));
  float* devin = new_gpu_array(N);
  float* devout = new_gpu_array(4*N);

  for(int i=0; i<N; i+=3){
    host1[i] = 1.;
  }
  memcpy_to_gpu(host1, devin, N);
  gpuFFT3dPlan_forward(plan, devin, devout);
  gpuFFT3dPlan_inverse(plan, devout, devin);
  



  
  
}