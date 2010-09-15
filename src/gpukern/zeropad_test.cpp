#include "gpu_mem.h"
#include "gpu_zeropad.h"
#include "timer.h"
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>

#define RUNS 1000

void format(float* array, int N1, int N2){
  for(int i=0; i<N1; i++){
    for(int j=0; j<N2; j++){
      printf("%f\t", array[i*N2+j]);
    }
    printf("\n");
  }
  printf("\n");
}

void test(){
  int N1 = 4, N2 = 5;
  int N = N1*N2;

  int M1 = N1, M2 = 2*N2;
  int M = M1 * M2;

  float* host = (float*)calloc(N, sizeof(float));
  float* dev = new_gpu_array(N);
  float* dev2 = new_gpu_array(M);
  float* host2 = (float*)calloc(M, sizeof(float));

  for(int i=0; i<N1; i++){
    for(int j=0; j<N2; j++){
      host[i*N2+j] = i + j / 100.;
    }
  }

  memcpy_to_gpu(host, dev, N);

  // do one to initialize CUDA before the actual timing
  gpu_copy_pad2D(dev, dev2, N1, N2, M1, M2);

  timer_start("zeropad");
  for(int i=0; i<RUNS; i++)
    gpu_copy_pad2D(dev, dev2, N1, N2, M1, M2);
  timer_stop("zeropad");

  memcpy_from_gpu(dev2, host2, N);
    format(host, N1, N2);
    format(host2, M1, M2);

  for(int i=0; i<N1; i++){
    for(int j=0; j<N2; j++){
      assert( host[i*N2+j] == host2[i*M2+j]);
    }
  }
  printf("PASS\n");
}

int main(){
  test();
}