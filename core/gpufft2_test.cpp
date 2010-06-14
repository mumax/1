/**
 * @file
 * Tests for smart zero-padded FFT on the GPU
 *
 * @author Arne Vansteenkiste
 * @author Ben Van de Wiele
 */

#include "gputil.h"
#include "gpufft2.h"
#include "gpuconv2.h"
#include "tensor.h"
#include "assert.h"

  int N0 = 1; // MUST ALSO WORK FOR 2D
  int N1 = 4;
  int N2 = 8;
  int N3 = 2; // real and imag part
  
void test_transpose(){

  int size[3] = {N0, N1, N2*N3};
    
  // (untransposed) "magnetization" on the host (CPU)
  tensor* mHost = new_tensor(3, N0, N1, N2*N3);
  
  // make some initial data
  float**** m = tensor_array4D(as_tensor(mHost->list, 4, N0, N1, N2, N3));
  for(int i=0; i<N0; i++)
    for(int j=0; j<N1; j++)
      for(int k=0; k<N2; k++){
    m[i][j][k][0] = i + j*0.01 + k*0.00001;
    m[i][j][k][1] = -( i + j*0.01 + k*0.00001 );
      }
  fprintf(stderr, "original:\n");
  format_tensor(mHost, stderr);
       
  // (untransposed) "magnetization" on the device (GPU)
  tensor* mDev = new_gputensor(3, size);
  tensor_copy_to_gpu(mHost, mDev);
  
  //________________________________________________________________ transpose YZ
  
  // transposed magnetization
  int sizeT[3] = {N0, N2, N1*N3};
  
  tensor* mDevT = new_gputensor(3, sizeT);
  tensor* mHostT = new_tensor(3, N0, N2, N1*N3); // N1 <-> N2
  
  gpu_tensor_transposeYZ_complex(mDev, mDevT);
  
  tensor_copy_from_gpu(mDevT, mHostT);
  format_tensor(mHostT, stderr);
  
  float**** mT = tensor_array4D(as_tensor(mHostT->list, 4, N0, N2, N1, N3)); // N1 <-> N2
  // test if it worked
  for(int i=0; i<N0; i++)
    for(int j=0; j<N1; j++)
      for(int k=0; k<N2; k++)
    for(int c=0; c<2; c++){
        assert(m[i][j][k][c] == mT[i][k][j][c]);    // j <-> k
    }
  
  //________________________________________________________________ transpose XZ
  
  int sizeT2[3] = {N2, N1, N0*N3};
  mDevT = new_gputensor(3, sizeT2);
  gpu_tensor_transposeXZ_complex(mDev, mDevT); 
  mHostT = new_tensor(3, N2, N1, N0*N3); // N0 <-> N2
  
  tensor_copy_from_gpu(mDevT, mHostT);
  format_tensor(mHostT, stderr);
  mT = tensor_array4D(as_tensor(mHostT->list, 4, N2, N1, N0, N3)); // N0 <-> N2
  // test if it worked
  for(int i=0; i<N0; i++)
    for(int j=0; j<N1; j++)
      for(int k=0; k<N2; k++)
    for(int c=0; c<2; c++){
        assert(m[i][j][k][c] == mT[k][j][i][c]); // i <-> k
    }
    
  fprintf(stderr, "PASS\n");
}

void test_fft(){

  int size[3]              = {N0, N1, N2};
  int kernelSize[3]        = {2*N0, 2*N1, 2*N2};
  int paddedStorageSize[3] = {kernelSize[X], kernelSize[Y], gpu_pad_to_stride(kernelSize[Z] + 2)};
  int size4D[4]              = {3, size[X], size[Y], size[Z]};
  int kernelSize4D[4]        = {3, kernelSize[X], kernelSize[Y], kernelSize[Z]};
  int paddedStorageSize4D[4] = {3, paddedStorageSize[X], paddedStorageSize[Y], paddedStorageSize[Z]};


  tensor* hostM = new_tensorN(4, size4D);
  tensor* hostMComp[3];
  for(int i=0; i<3; i++)
    hostMComp[i] = tensor_component(hostM, i);

  tensor* hostH = new_tensorN(4, size4D);
  tensor* hostHComp[3];
  for(int i=0; i<3; i++)
    hostHComp[i] = tensor_component(hostH, i);

  tensor* m = new_gputensor(4, size4D);
  tensor* mComp[3];
  for(int i=0; i<3; i++)
    mComp[i] = as_tensorN(NULL, 3, size);

  tensor* h = new_gputensor(4, size4D);
  tensor* hComp[3];
  for(int i=0; i<3; i++)
    hComp[i] = as_tensorN(NULL, 3, size);

  tensor* fft = new_gputensor(4, paddedStorageSize4D);
  tensor* fftComp[3];
  for(int i=0; i<3; i++)
    fftComp[i] = tensor_component(fft, i);

  gpuFFT3dPlan* plan = new_gpuFFT3dPlan_padded(size, kernelSize);

  float**** in = tensor_array4D(hostM);
  for(int c=0; c<3; c++)
  for(int i=0; i<N0; i++)
    for(int j=0; j<N1; j++)
      for(int k=0; k<N2; k++){
                in[c][i][j][k] = c + 1; //i + j*0.01 + k*0.00001;
      }
//   fprintf(stderr, "hostM:\n");
//   format_tensor(hostM, stderr);

  for(int i=0; i<3; i++){
    fprintf(stderr, "hostMComp[%d]:\n", i);
    format_tensor(hostMComp[i], stderr);
  }

  tensor_copy_to_gpu(hostM, m);
  fprintf(stderr, "m:\n");
  format_gputensor(m, stderr);

  for(int i=0; i<3; i++){
    mComp[i]->list = &(m->list[mComp[i]->len * i]);
    hComp[i]->list = &(h->list[hComp[i]->len * i]);
    fprintf(stderr, "m[%d]:\n", i);
    format_gputensor(mComp[i], stderr);
  }

  for(int i=0; i<3; i++){
    gpu_copy_pad(mComp[i], fftComp[i]);
  }

  fprintf(stderr, "fft:\n");
  format_gputensor(fft, stderr);

  for(int i=0; i<3; i++){
    gpuFFT3dPlan_forward(plan, fftComp[i], fftComp[i]);
  }

  for(int i=0; i<3; i++){
    gpuFFT3dPlan_inverse(plan, fftComp[i], fftComp[i]);
  }
  
  for(int i=0; i<3; i++){
    gpu_copy_unpad(fftComp[i], hComp[i]);
  }

  float N = kernelSize[X] * kernelSize[Y] * kernelSize[Z];
  
  tensor_copy_from_gpu(h, hostH);
  for(int i=0; i<hostH->len; i++){
    hostH->list[i] /= N;
  }

  format_tensor(hostH, stderr);
  
  float maxError = 0.0;
  for(int i=0; i<hostH->len; i++){
    if(fabs(hostM->list[i] - hostH->list[i]) > maxError){
      maxError = fabs(hostM->list[i] - hostH->list[i]);
    }
  }

  printf("FFT max error: %g\n", maxError);
  assert(maxError < 1E-5);
}

int main(int argc, char** argv){
  
  test_transpose();
  
  test_fft();
 
  return 0;
}