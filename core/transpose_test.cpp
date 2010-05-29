#include "gputil.h"
#include "gpuconv2.h"
#include "tensor.h"


int main(int argc, char** argv){
  int N0 = 1;
  int N1 = 3;
  int N2 = 4;
  int N3 = 2; // real and imag part
  
  tensor* mHost = new_tensor(3, N0, N1, N2*N3);
  int N = tensor_length(mHost);
  fprintf(stderr, "N=%d\n", N);
  
  float*** m = tensor_array3D(mHost);
  for(int i=0; i<N0; i++)
    for(int j=0; j<N1; j++)
      for(int k=0; k<N2*N3; k+=2){
	m[i][j][k] = i + j*0.01 + k*0.00001;
	m[i][j][k+1] = -( i + j*0.01 + k*0.00001 );
      }
  
  float* lDev = new_gpu_array(N);
  tensor* mDev = as_tensor(lDev, 4, N0, N1, N2, N3);
  float* lDevT = new_gpu_array(N);
  tensor* mDevT = as_tensor(lDevT, 4, N0, N1, N2, N3);
  tensor* mHostT = new_tensor(3, N0, N2, N1*N3);
  
  format_tensor(mHost, stderr);
  
  memcpy_to_gpu(mHost->list, mDev->list, N);
  
  gpu_transposeYZ_complex(mDev->list, mDevT->list, N0, N1, N2*N3); // complex !
  
  memcpy_from_gpu(mDevT->list, mHostT->list, N);
  
  format_tensor(mHostT, stderr);
  
  return 0;
}