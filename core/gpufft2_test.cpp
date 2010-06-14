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

  int N0 = 2;
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
       
  // (untransposed) "magnetization" on the device (gPU)
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

  int size[3] = {N0, N1, N2};
  int kernelSize[3] = {2*N0, 2*N1, 2*N2};
  gpuFFT3dPlan* plan = new_gpuFFT3dPlan_padded(size, kernelSize);

    
// make some host data and initialize _________________________________
  tensor* Host_in = new_tensor(3, N0, N1, N2);
  int N = tensor_length(Host_in);

  float*** in = tensor_array3D(Host_in);
  for(int i=0; i<N0; i++)
    for(int j=0; j<N1; j++)
      for(int k=0; k<N2; k++){
                in[i][j][k] = i + j*0.01 + k*0.00001;
      }
  fprintf(stderr, "original:\n");
  format_tensor(Host_in, stderr);
// _____________________________________________________________________

    
// copy host data in zero-padded tensor ________________________________
  tensor* Host_padded_in = new_tensor(3, plan->paddedStorageSize[X], plan->paddedStorageSize[Y], plan->paddedStorageSize[Z]);
  int N_padded = tensor_length(Host_padded_in);

  float*** padded_in = tensor_array3D(Host_padded_in);
  for(int i=0; i<N0; i++)
    for(int j=0; j<N1; j++)
      for(int k=0; k<N2; k++){
                padded_in[i][j][k] = in[i][j][k];
      }

    fprintf(stderr, "original, padded:\n");
  format_tensor(Host_padded_in, stderr);
// _____________________________________________________________________
  
// make data on device__________________________________________________
  tensor* Dev_in = as_tensor(new_gpu_array(N_padded), 3, plan->paddedStorageSize[X], plan->paddedStorageSize[Y], plan->paddedStorageSize[Z]);
  tensor* Dev_out = Dev_in; // in-place
  
  //tensor* Dev_out = as_tensor(new_gpu_array(N_padded), 3, plan->paddedStorageSize[X], plan->paddedStorageSize[Y], plan->paddedStorageSize[Z]); // out of place is broken now
  
  memcpy_to_gpu(Host_padded_in->list, Dev_in->list, N_padded);
// _____________________________________________________________________
       

    tensor* Host_padded_out = new_tensor(3, plan->paddedStorageSize[X], plan->paddedStorageSize[Y], plan->paddedStorageSize[Z]);

// test forward ________________________________________________________
  gpuFFT3dPlan_forward(plan, Dev_in, Dev_out);
  gpu_zero_tensor(Dev_in);
  
  memcpy_from_gpu(Dev_in->list, Host_padded_out->list, N_padded);
  fprintf(stderr, "\n\nforward:\n");
  format_tensor(Host_padded_out, stderr);
// _____________________________________________________________________

    
// // test inverse ________________________________________________________
  gpuFFT3dPlan_inverse(plan, Dev_out, Dev_in);
  memcpy_from_gpu(Dev_in->list, Host_padded_out->list, N_padded);
  fprintf(stderr, "\n\ninverse:\n");
  format_tensor(Host_padded_out, stderr);
// _____________________________________________________________________


// copy result to unpadded tensor ______________________________________    
  tensor* Host_out = new_tensor(3, N0, N1, N2);
  float*** padded_out = tensor_array3D(Host_padded_out);

  float*** out = tensor_array3D(Host_out);
  for(int i=0; i<N0; i++)
    for(int j=0; j<N1; j++)
      for(int k=0; k<N2; k++){
                out[i][j][k] = padded_out[i][j][k]/(float)plan->paddedN;
      }
  fprintf(stderr, "Output:\n");
  format_tensor(Host_out, stderr);
// _____________________________________________________________________

// compare input <-> output after forward and inverse transform ________
  int error = 0;
  
  for(int i=0; i<N0; i++)
    for(int j=0; j<N1; j++)
      for(int k=0; k<N2; k++){
                if ( (in[i][j][k] - out[i][j][k]) > 1e-4){
                    fprintf(stderr, "error element: %d, %d, %d\n", i, j, k );
                    error = 1;
                }
      }
// _____________________________________________________________________

    if(error == 0)
      fprintf(stderr, "PASS\n");
    else{
      fprintf(stderr, "FAIL\n");
      exit(error);
    }
  
}

int main(int argc, char** argv){
  
  test_transpose();
  
  test_fft();
 
  return 0;
}