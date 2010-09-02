#include "gpu_mem.h"
#include "gpu_transpose.h"
#include "timer.h"
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
  int N1 = 256, N2 = 512;
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
  
  timer_start("transpose");
  for(int i=0; i<10000; i++)
    gpu_transpose(dev, dev2, N1, N2);
  timer_stop("transpose");
  
  memcpy_from_gpu(dev2, host2, N);
//   format(host, N1, N2);
//   format(host2, N2, N1);

  for(int i=0; i<N1; i++){
    for(int j=0; j<N2; j++){
      assert( host[i*N2+j] == host2[j*N1+i]);
    }
  }

  timer_printdetail();

  printf("PASS\n");
}