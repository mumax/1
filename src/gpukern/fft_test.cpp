#include "gpukern.h"
#include "timer.h"
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>



void printt(char* tag, float* data, int N0, int N1, int N2){
  printf("%s\n", tag);
  for(int i=0; i<N0; i++){
    for(int j=0; j<N1; j++){
      for(int k=0; k<N2; k++){
        fprintf(stdout, "%g\t", data[i*N1*N2 + j*N2 + k]);
      }
      printf("\n");
    }
    printf("\n");
  }
}

int main(){
  int N0=2, N1=4, N2=8;
  int N = N0*N1*N2;
  int* size = new int[3];
  size[0] = N0;
  size[1] = N1;
  size[2] = N2;
  printf("%d %d %d\n", size[0], size[1], size[2]);
  
  int* paddedsize = new int[3];
  paddedsize[0] = 2*size[0];
  paddedsize[1] = 2*size[1];
  paddedsize[2] = 2*size[2];
  int paddedN = paddedsize[0] * paddedsize[1] * paddedsize[2];
  printf("%d %d %d\n", paddedsize[0], paddedsize[1], paddedsize[2]);

  
  gpuFFT3dPlan* plan = new_gpuFFT3dPlan_padded(size, paddedsize);
    
  float* host1 = (float*)calloc(8*paddedN, sizeof(float));
  float* host2 = (float*)calloc(8*paddedN, sizeof(float));
  float* devin = new_gpu_array(8*paddedN);
  float* devout = new_gpu_array(8*paddedN);

    for(int i = 0; i < size[0]; i++ ){
      for(int j = 0; j < size[1]; j++ ){
        for(int k = 0; k < size[2]; k++ ){
          host1[i*paddedsize[1]*paddedsize[2] + j*paddedsize[2] + k] = 1.; i*paddedsize[1]*paddedsize[2]+j*paddedsize[2]+k;
//           host1[i*size[1]*size[2]+j*size[2]+k] = i*size[1]*size[2]+j*size[2]+k;
        }
      }
    }

  for(int i=0; i<paddedN; i++){
    host1[i] = 1.;
  }


  printt("host1", host1, paddedsize[0], paddedsize[1], paddedsize[2]);
  memcpy_to_gpu(host1, devin, N);

    
  gpuFFT3dPlan_forward(plan, devin, devout);
  gpuFFT3dPlan_inverse(plan, devout, devin);

  
}