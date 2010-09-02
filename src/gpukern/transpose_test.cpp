#include "gpu_mem.h"
#include "gpu_transpose.h"
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>

void format(float* array, int N1, int N2){
  for(int i=0; i<N1; i++){
    for(int j=0; j<N2; j++){
      printf("%f\t", array[i*N2+j]);
    }
    printf("\n");
  }
  printf("\n");
}

int main(){
  int N1 = 65, N2 = 31;
  int N = N1*N2;
  
  float* host = (float*)calloc(N, sizeof(float));
  float* dev = new_gpu_array(N);
  float* dev2 = new_gpu_array(N);
  float* host2 = (float*)calloc(N, sizeof(float));

  for(int i=0; i<N1; i++){
    for(int j=0; j<N2; j++){
      host[i*N2+j] = i + j / 100.;
    }
  }

  memcpy_to_gpu(host, dev, N);
  gpu_transpose(dev, dev2, N1, N2);
  memcpy_from_gpu(dev2, host2, N);
  format(host, N1, N2);
  format(host2, N2, N1);

  for(int i=0; i<N1; i++){
    for(int j=0; j<N2; j++){
      assert( host[i*N2+j] == host2[j*N1+i]);
    }
  }

  printf("PASS\n");
}