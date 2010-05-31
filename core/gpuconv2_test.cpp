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
  
  // make fft plan
  int* zero_pad = new int[3];
  zero_pad[X] = 1;
  zero_pad[X] = 0;
  zero_pad[X] = 0;
  gpu_plan3d_real_input* plan = new_gpu_plan3d_real_input(N0, N1, N2, zero_pad);

	
// make some host data and initialize _________________________________
  tensor* mHost = new_tensor(3, N0, N1, N2);
  int N = tensor_length(mHost);

	float*** m = tensor_array3D(mHost);
  for(int i=0; i<N0; i++)
    for(int j=0; j<N1; j++)
      for(int k=0; k<N2; k++){
//				m[i][j][k] = i + j*0.01 + k*0.00001;
				m[i][j][k] = 1.0f;
      }
  fprintf(stderr, "original:\n");
  format_tensor(mHost, stderr);
// _____________________________________________________________________

	
// copy host data in zero-padded tensor ________________________________
  tensor* mHost_padded = new_tensor(3, plan->paddedStorageSize[X], plan->paddedStorageSize[Y], plan->paddedStorageSize[Z]);
  int N_padded = tensor_length(mHost_padded);

	float*** m_padded = tensor_array3D(mHost_padded);
  for(int i=0; i<N0; i++)
    for(int j=0; j<N1; j++)
      for(int k=0; k<N2; k++){
				m_padded[i][j][k] = m[i][j][k];
      }

	fprintf(stderr, "original, padded:\n");
  format_tensor(mHost_padded, stderr);
// _____________________________________________________________________
  
// makeata on device____________________________________________________
  tensor* mDev = as_tensor(new_gpu_array(N_padded), 3, plan->paddedStorageSize[X], plan->paddedStorageSize[Y], plan->paddedStorageSize[Z]);
  memcpy_to_gpu(mHost_padded->list, mDev->list, N_padded);
// _____________________________________________________________________
       

	tensor* output_padded = new_tensor(3, plan->paddedStorageSize[X], plan->paddedStorageSize[Y], plan->paddedStorageSize[Z]);

// test forward ________________________________________________________
  gpu_plan3d_real_input_forward(plan, mDev->list);
  memcpy_from_gpu(mDev->list, output_padded->list, N_padded);
  format_tensor(output_padded, stderr);

// _____________________________________________________________________

	
//   // test inverse
/*  gpu_plan3d_real_input_inverse(plan, mDev->list);
  memcpy_from_gpu(mDev->list, mHost->list, N);
  format_tensor(mHost, stderr);
  
  fprintf(stderr, "\n\ninverse:\n");*/
  
  //fprintf(stderr, "PASS\n");
  return 0;
}