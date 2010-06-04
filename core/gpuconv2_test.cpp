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

  int N0 = 1;
  int N1 = 4;
  int N2 = 2;

  tensor* m = new_tensor(3, N0, N1, N2);
  for(int i=0; i<tensor_length(m); i++){
    m->list[i] = i;
  }
  format_tensor(m, stderr);
  
  int* zero_pad = new int[3];
  zero_pad[X] = zero_pad[Y] = zero_pad[Z] = 1;
  
  tensor* kernel = new_tensor(5, 3, 3, 2*N0, 2*N1, 2*N2);
  gpuconv2* conv = new_gpuconv2(N0, N1, N2, kernel, zero_pad);

  fprintf(stderr, "PASS\n");
  return 0;
}