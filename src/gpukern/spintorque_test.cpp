#include "gpu_mem.h"
#include "gpu_spintorque.h"
#include "timer.h"
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>

#define RUNS 100

void format(float* array, int N0, int N1, int N2){
  for(int x=0; x<N0; x++)
  for(int i=0; i<N1; i++){
    for(int j=0; j<N2; j++){
      printf("%f\t", array[x*N1*N2 + i*N2 + j]);
    }
    printf("\n");
  }
  printf("\n");
}

int main(){

  int N0 = 1, N1 = 16, N2 = 32;
  int N = N0*N1*N2;

  float* host = (float*)calloc(N, sizeof(float));
  float* dev = new_gpu_array(N);

  float* dev2 = new_gpu_array(N);
  float* host2 = (float*)calloc(N, sizeof(float));

  for(int x=0; x<N0; x++){
    for(int i=0; i<N1; i++){
      for(int j=0; j<N2; j++){
        host[x*N1*N2 + i*N2 + j] = 10*x + i + j / 100.;
      }
    }
  }
  format(host, N0, N1, N2);
  memcpy_to_gpu(host, dev, N);

  gpu_directionial_diff(1, 0, 0, dev, dev2, N0, N1, N2);
  
  memcpy_from_gpu(dev2, host2, N);
  //   format(host, N1, N2);
  //   format(host2, N2, N1);

  format(host2, N0, N1, N2);

  for(int i=0; i<N; i++){
    assert(host[i] == host2[i]);
  }
  printf("PASS\n");
  
  timer_printdetail();
}