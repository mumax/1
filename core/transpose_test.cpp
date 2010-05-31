/**
 * @file
 * Tests for transposing complex number arrays on the GPU
 *
 * @author Arne Vansteenkiste
 */

#include "timer.h"
#include "gputil.h"
#include "gpuconv2.h"
#include "tensor.h"
#include "assert.h"

int main(int argc, char** argv){
  
  int N0 = 64;
  int N1 = 256;
  int N2 = 256;
  int N3 = 2; // real and imag part
  
  // (untransposed) "magnetization" on the host (CPU)
  tensor* mHost = new_tensor(3, N0, N1, N2*N3);
  int N = tensor_length(mHost);
  // make some initial data
  float**** m = tensor_array4D(as_tensor(mHost->list, 4, N0, N1, N2, N3));
  for(int i=0; i<N0; i++)
    for(int j=0; j<N1; j++)
      for(int k=0; k<N2; k++){
	m[i][j][k][0] = i + j*0.01 + k*0.00001;
	m[i][j][k][1] = -( i + j*0.01 + k*0.00001 );
      }
  fprintf(stderr, "original:\n");
//   format_tensor(mHost, stderr);
       
  // (untransposed) "magnetization" on the device (gPU)
  float* lDev = new_gpu_array(N);
  tensor* mDev = as_tensor(lDev, 4, N0, N1, N2, N3);
  memcpy_to_gpu(mHost->list, mDev->list, N);
  
  //________________________________________________________________ transpose YZ
  
  // transposed magnetization
  float* lDevT = new_gpu_array(N);
  tensor* mHostT = new_tensor(3, N0, N2, N1*N3); // N1 <-> N2
  
  gpu_transposeYZ_complex(mDev->list, lDevT, N0, N1, N2*N3); // complex !
  
  memcpy_from_gpu(lDevT, mHostT->list, N);
//   format_tensor(mHostT, stderr);
  float**** mT = tensor_array4D(as_tensor(mHostT->list, 4, N0, N2, N1, N3)); // N1 <-> N2
  // test if it worked
  for(int i=0; i<N0; i++)
    for(int j=0; j<N1; j++)
      for(int k=0; k<N2; k++)
	for(int c=0; c<2; c++){
	    assert(m[i][j][k][c] == mT[i][k][j][c]);	// j <-> k
	}
  
  //________________________________________________________________ transpose XZ
  
  gpu_transposeXZ_complex(mDev->list, lDevT, N0, N1, N2*N3); // complex !
  mHostT = new_tensor(3, N2, N1, N0*N3); // N0 <-> N2
  
  memcpy_from_gpu(lDevT, mHostT->list, N);
//   format_tensor(mHostT, stderr);
  mT = tensor_array4D(as_tensor(mHostT->list, 4, N2, N1, N0, N3)); // N0 <-> N2
  // test if it worked
  for(int i=0; i<N0; i++)
    for(int j=0; j<N1; j++)
      for(int k=0; k<N2; k++)
	for(int c=0; c<2; c++){
	    assert(m[i][j][k][c] == mT[k][j][i][c]); // i <-> k
	}
	
  timer_printdetail();
  fprintf(stderr, "PASS\n");
  return 0;
}