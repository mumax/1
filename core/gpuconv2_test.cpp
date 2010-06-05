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
  tensor* mDev = new_gputensor(4, m->size);
  memcpy_to_gpu(m->list, mDev->list, m->len);
  
  tensor* kernel = new_tensor(5, 3, 3, paddedSize[X], paddedSize[Y], paddedSize[Z]);
  gpuconv2* conv = new_gpuconv2(size, paddedSize);
  gpuconv2_loadkernel5DSymm(conv, kernel);
  
  tensor* h = new_tensorN(4, m->size);
  tensor* hDev = new_gputensor(4, h->size);
  for(int i=0; i<3; i++)
    assert(h->size[i] == m->size[i]);
  
  gpuconv2_exec(conv, mDev, hDev);
  memcpy_from_gpu(hDev->list, h->list, h->len);
  format_tensor(h, stderr);

  fprintf(stderr, "PASS\n");
  return 0;
}