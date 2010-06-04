/**
 * @file
 * Tests for smart zero-padded FFT and convolution on the GPU
 *
 * @author Arne Vansteenkiste, Ben Van de Wiele
 */

#include "gputil.h"
#include "gpuconv2.h"
#include "tensor.h"
#include "assert.h"

int main(int argc, char** argv){
  fprintf(stderr, "gpuconv2_test\n");

  int size[3] = {1, 2, 4};
  int paddedSize[3] = {2*size[X], 2*size[Y], 2*size[Z]};
  

  tensor* m = new_tensor(4, 3, size[X], size[Y], size[Z]);
  for(int i=0; i<tensor_length(m); i++){
    m->list[i] = i;
  }
  format_tensor(m, stderr);
  
  int* zero_pad = new int[3];
  zero_pad[X] = zero_pad[Y] = zero_pad[Z] = 1;
  
  tensor* kernel = new_tensor(5, 3, 3, paddedSize[X], paddedSize[Y], paddedSize[Z]);
  gpuconv2* conv = new_gpuconv2(size, paddedSize);
  gpuconv2_loadkernel5DSymm(conv, kernel);

  fprintf(stderr, "PASS\n");
  return 0;
}