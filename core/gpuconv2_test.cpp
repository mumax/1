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

  int N0 = 2;
  int N1 = 4;
  int N2 = 6;
  
  
void test_pad(){
  
  int size[3] = {N0, N1, N2};
  int paddedSize[3] = {2*size[X], 2*size[Y], 2*size[Z]};
  
  tensor* small = new_tensorN(3, size);
  for(int i=0; i<small->len; i++){
    small->list[i] = i;
  }
  tensor* smallDev = new_gputensor(3, size);
  tensor_copy_to_gpu(small, smallDev);
  format_tensor(small, stderr);
  
  tensor* large = new_tensor(3, 2*N0, 2*N1, 2*N2);
  tensor* largeDev = new_gputensor(3, paddedSize);
  
  gpu_copy_pad(smallDev, largeDev);
  
  tensor_copy_from_gpu(largeDev, large);
  format_tensor(large, stderr);
  
  gpu_zero_tensor(smallDev);
  tensor_zero(small);
  
  gpu_copy_unpad(largeDev, smallDev);
  tensor_copy_from_gpu(smallDev, small);
  format_tensor(small, stderr);
  
}

int main(int argc, char** argv){
  fprintf(stderr, "gpuconv2_test\n");

  test_pad();
  return 0;
  
  int size[3] = {N0, N1, N2};
  int paddedSize[3] = {2*size[X], 2*size[Y], 2*size[Z]};

  tensor* m = new_tensor(4, 3, size[X], size[Y], size[Z]);
  for(int i=0; i<tensor_length(m); i++){
    m->list[i] = 1.;
  }
  format_tensor(m, stderr);
  tensor* mDev = new_gputensor(4, m->size);
  memcpy_to_gpu(m->list, mDev->list, m->len);
  
  tensor* kernel = new_tensor(5, 3, 3, paddedSize[X], paddedSize[Y], paddedSize[Z]);
  gpuconv2* conv = new_gpuconv2(size, paddedSize);
  gpuconv2_loadkernel5DSymm(conv, kernel);
  
  tensor* h = new_tensorN(4, m->size);
  for(int i=0; i<h->len; i++)
    h->list[i]= 1.;
  
  tensor* hDev = new_gputensor(4, h->size);
  tensor_copy_to_gpu(h, hDev);
  
  gpuconv2_exec(conv, mDev, hDev);
  memcpy_from_gpu(hDev->list, h->list, h->len);
  
  for(int i=0; i<h->len; i++)
    h->list[i] /= (float)(size[X] * size[Y] * size[Z]);
  
  format_tensor(h, stderr);

  fprintf(stderr, "PASS\n");
  return 0;
}