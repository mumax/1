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

void test_complex(){
  int N1 = 256, N2 = 512, N3=2;
  int N = N1*N2*N3;

  float* host = (float*)calloc(N, sizeof(float));
  float* dev = new_gpu_array(N);

  printf("%p\n", dev);
  
  float* dev2 = new_gpu_array(N);
  float* host2 = (float*)calloc(N, sizeof(float));

  for(int i=0; i<N1; i++){
    for(int j=0; j<N2; j++){
      host[i*N2*N3 + j*N3 + 0] = i + j / 100.;
      host[i*N2*N3 + j*N3 + 1] = - (i + j / 100.);
    }
  }

  memcpy_to_gpu(host, dev, N);

  // do one to initialize CUDA before the actual timing
  gpu_transpose_complex(dev, dev2, N1, N2*N3);

  timer_start("transpose_complex");
  for(int i=0; i<10000; i++)
    gpu_transpose_complex(dev, dev2, N1, N2*N3);
  timer_stop("transpose_complex");

  memcpy_from_gpu(dev2, host2, N);
//   format(host, N1, N2*N3);
//   format(host2, N2, N1*N3);

  for(int i=0; i<N1; i++){
    for(int j=0; j<N2; j++){
      assert( host[i*N2*N3 + j*N3 + 0] == host2[j*N1*N3 + i*N3 + 0] );
      assert( host[i*N2*N3 + j*N3 + 1] == host2[j*N1*N3 + i*N3 + 1] );
    }
  }

  printf("PASS\n");
}

void test_real(){
  int N1 = 256, N2 = 1024;
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

  // do one to initialize CUDA before the actual timing
  gpu_transpose(dev, dev2, N1, N2);

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
  printf("PASS\n");
}

int main(){
  test_real();
   test_complex();
    timer_printdetail();
}