/**
 * @file
 * Tests for smart zero-padded FFT and convolution on the GPU
 *
 * @author Arne Vansteenkiste
 */

#include "gputil.h"
#include "gpuconv2.h"
#include "tensor.h"
#include "assert.h"

int main(int argc, char** argv){
  
  int N0 = 2;
  int N1 = 4;
  int N2 = 8;
  
  // make some host data
  
  tensor* mHost = new_tensor(3, N0, N1, N2);
  int N = tensor_length(mHost);
  
  // make some initial data
  float*** m = tensor_array3D(mHost);
  for(int i=0; i<N0; i++)
    for(int j=0; j<N1; j++)
      for(int k=0; k<N2; k++){
//				m[i][j][k] = i + j*0.01 + k*0.00001;
				m[i][j][k] = 1.0f;
      }
//  mHost->list[0] = 1.0;
  fprintf(stderr, "original:\n");
  format_tensor(mHost, stderr);
  
  // make data on device
  tensor* mDev = as_tensor(new_gpu_array(N), 3, N0, N1, N2);
  memcpy_to_gpu(mHost->list, mDev->list, N);
       
  // make fft plan
  int* zero_pad = new int[3];
  zero_pad[X] = 0;
  zero_pad[X] = 0;
  zero_pad[X] = 0;
  gpu_plan3d_real_input* plan = new_gpu_plan3d_real_input(N0, N1, N2, zero_pad);
  
  // test forward
  gpu_plan3d_real_input_forward(plan, mDev->list);
  memcpy_from_gpu(mDev->list, mHost->list, N);
  format_tensor(mHost, stderr);
  
//   // test backward
  gpu_plan3d_real_input_inverse(plan, mDev->list);
  memcpy_from_gpu(mDev->list, mHost->list, N);
  format_tensor(mHost, stderr);
  
  fprintf(stderr, "\n\ninverse:\n");
  
  //fprintf(stderr, "PASS\n");
  return 0;
}