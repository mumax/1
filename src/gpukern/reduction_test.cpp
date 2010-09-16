#include "gpu_mem.h"
#include "gpu_reduction.h"
#include "timer.h"
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>

#define divUp(x, y) ( (((x)-1)/(y)) +1 )

void test_reduction(){

//   int* Ns = int{1, 2, 3, 8, 16, 17, 31, 32, 33, 64, 65, 127, 128, 129, 255, 256, 257, 512, 1024, 2048, 8192}
//   for(int i=0; i<21; i++){
//     int N = Ns[i];

  int N = 1024;

  float* host = (float*)calloc(N, sizeof(float));
  float* dev1 = new_gpu_array(N);



    for(int i=0; i<N; i++){
      host[i] = 1.;
    }

    memcpy_to_gpu(host, dev1, N);

    float sum = gpu_sum(dev1, N);

    assert(sum == N);
    printf("PASS\n");
//   }
}



int main(){
  test_reduction();
}