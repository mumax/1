/*
 *  This file is part of MuMax, a high-performance micromagnetic simulator.
 *  Copyright 2010  Arne Vansteenkiste, Ben Van de Wiele.
 *  Use of this source code is governed by the GNU General Public License version 3
 *  (as published by the Free Software Foundation) that can be found in the license.txt file.
 *
 *  Note that you are welcome to modify this code under condition that you do not remove any 
 *  copyright notices and prominently state that you modified it, giving a relevant date.
 */

#include "tensor.h"
#include "gputil.h"
#include "gpu_fft6.h"
#include "gpu_fftbig.h"
#include "assert.h"
#include "timer.h"
#include <stdio.h>
#include "gpu_conf.h"
#include "gpu_kernel_biot_savart3d.h"

#ifdef __cplusplus
extern "C" {
#endif

#define BLOCKSIZE 16


// tensor *gpu_micromag3d_kernel(int *kernelSize, float *cellsize, int exchType, int *exchInConv, int *repetition){
void gpu_kernel_biot_savart3d(int *kernelSize, float *cellsize, int *repetition){
  
  // check input + allocate tensor on device ______________________________________________________
    int kernelStorageN = kernelSize[X] * kernelSize[Y] * (kernelSize[Z]+2);

    tensor *dev_kernel;
    if (p->size[X]==1)
      dev_kernel = as_tensor(new_gpu_array(2*kernelStorageN/2), 2, 2, kernelStorageN/2);  // only real parts!!
    else
      dev_kernel = as_tensor(new_gpu_array(3*kernelStorageN/2), 2, 3, kernelStorageN/2);  // only real parts!!
  // ______________________________________________________________________________________________


  // initialization Gauss quadrature points for integrations + copy to gpu ________________________
    float *dev_qd_W_10 = new_gpu_array(10);
    float *dev_qd_P_10 = new_gpu_array(3*10);
    initialize_Gauss_quadrature_on_gpu_biot_savart3d(dev_qd_W_10, dev_qd_P_10, cellSize);
  // ______________________________________________________________________________________________
  
  
  // Plan initialization of FFTs and initialization of the kernel _________________________________
    gpuFFT3dPlan* kernel_plan = new_gpuFFT3dPlan_padded(kernelSize, kernelSize);
    gpu_init_and_FFT_Greens_kernel_elements_biot_savart3d(dev_kernel->list, kernelSize, cellSize, repetition, dev_qd_P_10, dev_qd_W_10, kernel_plan);
  // ______________________________________________________________________________________________ 
  
  delete_FFT3dPlan(kernel_plan);
  cudaFree (dev_qd_W_10);
  cudaFree (dev_qd_P_10);

  write_tensor(tensor* t, FILE* out);
  return;
//  return (dev_kernel);   ///> moet de kernel hier gefreed worden?
}



/// remark: number of FD cells in a dimension can not be odd if no zero padding!!
void gpu_init_and_FFT_Greens_kernel_elements_biot_savart3d(float *dev_kernel, int *kernelSize, float *FD_cell_size, int *repetition, float *dev_qd_P_10, float *dev_qd_W_10, gpuFFT3dPlan* kernel_plan){

  
  int kernelN = kernelSize[X]*kernelSize[Y]*kernelSize[Z];                              // size of a kernel component without zeros
  float *dev_temp1 = new_gpu_array(kernelN);                                            // temp array on device for storage of kernel component without zeros (input of fft routine)
  int kernelStorageN = kernelSize[X]*kernelSize[Y]*(kernelSize[Z]+2);                   // size of a zero padded kernel component
  float *dev_temp2 = new_gpu_array(kernelStorageN);                                     // temp array on device for storage of zero padded kernel component (output of fft routine)
  
  // Define gpugrids and blocks ___________________________________________________________________

    dim3 gridsize1((kernelSize[X]+1)/2,kernelSize[Y]/2, 1);
    dim3 blocksize1(1,1,1);
    check3dconf(gridsize1, blocksize1);

    int N2 = kernelStorageN/2;
    dim3 gridsize2, blocksize2;
    make1dconf(N2, &gridsize2, &blocksize2);
  // ______________________________________________________________________________________________
  
  // Main function operations _____________________________________________________________________
    int rank0 = 0;                         // defines the first rank of the biot savart kernel, N0>1: [x, y, z], N1=1: [y, z]
    for (int co=0; co<3; co++){            // for a Biot Savart kernel component [X, Y, Z]:
      if (co==0 && kernelSize[0]==1)  continue;    // N0=1 -> x component has only zeros, so left out.
        // Put all elements in 'dev_temp1' to zero.
      gpu_zero(dev_temp1, kernelN);    
      gpu_sync();
        // Fill in the elements.
      _gpu_init_Greens_kernel_elements_biot_savart3d<<<gridsize1, blocksize1>>>(dev_temp1, kernelSize[X], kernelSize[Y], kernelSize[Z], co, FD_cell_size[X], FD_cell_size[Y], FD_cell_size[Z], repetition[X], repetition[Y], repetition[Z], dev_qd_P_10, dev_qd_W_10);
      gpu_sync();
        // Fourier transform the kernel component.
      gpuFFT3dPlan_forward(kernel_plan, dev_temp1, dev_temp2);
      gpu_sync();
        // Copy the real parts to the corresponding place in the dev_kernel tensor.
      _gpu_extract_real_parts_biot_savart3d<<<gridsize2, blocksize2>>>(&dev_kernel[rank0*kernelStorageN/2], dev_temp2, N2);
      gpu_sync();
      rank0++;                                        // get ready for next component
    } 
  // ______________________________________________________________________________________________


  cudaFree (dev_temp1);
  cudaFree (dev_temp2);
  
  return;
}


__global__ void _gpu_init_Greens_kernel_elements_micromag3d(float *dev_temp, int Nkernel_X, int Nkernel_Y, int Nkernel_Z, int co, float FD_cell_size_X, float FD_cell_size_Y, float FD_cell_size_Z, int repetition_X, int repetition_Y, int repetition_Z, float *dev_qd_P_10, float *dev_qd_W_10){

  /// @todo possible redeclaration of threadparameters required when using 'make3dconf' for thread launching.

  int i = blockIdx.x;
  int j = blockIdx.y;

  int N2 = Nkernel_Z;
  int N12 = Nkernel_Y * N2;

  if ( i<((Nkernel_X+1)/2) && j<(Nkernel_Y/2) )
    for (int k=0; k<N2/2; k++){
        dev_temp[            i*N12 +             j*N2 +           k] = _gpu_get_Greens_element_biot_savart3d(Nkernel_X, Nkernel_Y, Nkernel_Z, co,  i,  j,  k, FD_cell_size_X, FD_cell_size_Y, FD_cell_size_Z, repetition_X, repetition_Y, repetition_Z, dev_qd_P_10, dev_qd_W_10);
      if (i>0)
        dev_temp[(Nkernel_X-i)*N12 +             j*N2 +           k] = _gpu_get_Greens_element_biot_savart3d(Nkernel_X, Nkernel_Y, Nkernel_Z, co, -i,  j,  k, FD_cell_size_X, FD_cell_size_Y, FD_cell_size_Z, repetition_X, repetition_Y, repetition_Z, dev_qd_P_10, dev_qd_W_10);
      if (j>0)
        dev_temp[            i*N12 + (Nkernel_Y-j)*N2 +           k] = _gpu_get_Greens_element_biot_savart3d(Nkernel_X, Nkernel_Y, Nkernel_Z, co,  i, -j,  k, FD_cell_size_X, FD_cell_size_Y, FD_cell_size_Z, repetition_X, repetition_Y, repetition_Z, dev_qd_P_10, dev_qd_W_10);
      if (k>0) 
        dev_temp[            i*N12 +             j*N2 + Nkernel_Z-k] = _gpu_get_Greens_element_biot_savart3d(Nkernel_X, Nkernel_Y, Nkernel_Z, co,  i,  j, -k, FD_cell_size_X, FD_cell_size_Y, FD_cell_size_Z, repetition_X, repetition_Y, repetition_Z, dev_qd_P_10, dev_qd_W_10);
      if (i>0 && j>0)
        dev_temp[(Nkernel_X-i)*N12 + (Nkernel_Y-j)*N2 +           k] = _gpu_get_Greens_element_biot_savart3d(Nkernel_X, Nkernel_Y, Nkernel_Z, co, -i, -j,  k, FD_cell_size_X, FD_cell_size_Y, FD_cell_size_Z, repetition_X, repetition_Y, repetition_Z, dev_qd_P_10, dev_qd_W_10);
      if (i>0 && k>0) 
        dev_temp[(Nkernel_X-i)*N12 +             j*N2 + Nkernel_Z-k] = _gpu_get_Greens_element_biot_savart3d(Nkernel_X, Nkernel_Y, Nkernel_Z, co, -i,  j, -k, FD_cell_size_X, FD_cell_size_Y, FD_cell_size_Z, repetition_X, repetition_Y, repetition_Z, dev_qd_P_10, dev_qd_W_10);
      if (j>0 && k>0) 
        dev_temp[            i*N12 + (Nkernel_Y-j)*N2 + Nkernel_Z-k] = _gpu_get_Greens_element_biot_savart3d(Nkernel_X, Nkernel_Y, Nkernel_Z, co,  i, -j, -k, FD_cell_size_X, FD_cell_size_Y, FD_cell_size_Z, repetition_X, repetition_Y, repetition_Z, dev_qd_P_10, dev_qd_W_10);
      if (i>0 && j>0 && k>0) 
        dev_temp[(Nkernel_X-i)*N12 + (Nkernel_Y-j)*N2 + Nkernel_Z-k] = _gpu_get_Greens_element_biot_savart3d(Nkernel_X, Nkernel_Y, Nkernel_Z, co, -i, -j, -k, FD_cell_size_X, FD_cell_size_Y, FD_cell_size_Z, repetition_X, repetition_Y, repetition_Z, dev_qd_P_10, dev_qd_W_10);
    }
  
  return;
}

__device__ float _gpu_get_Greens_element_biot_savart3d(int Nkernel_X, int Nkernel_Y, int Nkernel_Z, int co, int a, int b, int c, float FD_cell_size_X, float FD_cell_size_Y, float FD_cell_size_Z, int repetition_X, int repetition_Y, int repetition_Z, float *dev_qd_P_10, float *dev_qd_W_10){

  float result = 0.0f;
  float *dev_qd_P_10_X = &dev_qd_P_10[X*10];
  float *dev_qd_P_10_Y = &dev_qd_P_10[Y*10];
  float *dev_qd_P_10_Z = &dev_qd_P_10[Z*10];
  float dim_inverse = 1.0f/( (float) Nkernel_X*Nkernel_Y*Nkernel_Z );
  
  // for elements in Kernel component x ___________________________________________________________
    if (co==0){

      for(int cnta=-repetition_X; cnta<=repetition_X; cnta++)
      for(int cntb=-repetition_Y; cntb<=repetition_Y; cntb++)
      for(int cntc=-repetition_Z; cntc<=repetition_Z; cntc++){

        int i = a + cnta*Nkernel_X/2;
        int j = b + cntb*Nkernel_Y/2;
        int k = c + cntc*Nkernel_Z/2;
        int r2_int = i*i+j*j+k*k;

        if (r2_int<400){
          for (int=cnt1; cnt1<10; cnt1++){
            x = i*FD_cell_size_X + dev_qd_P_10_X[cnt1];
            for (int cnt2=0; cnt2<10; cnt2++){
              float y = j * FD_cell_size_Y + dev_qd_P_10_Y[cnt2];
              for (int cnt3=0; cnt3<10; cnt3++){
                float z = k * FD_cell_size_Z + dev_qd_P_10_Z[cnt3];
                result += FD_cell_size_X * FD_cell_size_Y * FD_cell_size_Z / 8.0f 
                          * dev_qd_W_10[cnt1] * dev_qd_W_10[cnt2] * dev_qd_W_10[cnt3]
                          * ( x*__powf(x*x+y*y+z*z, -1.5f) );
              }
            }
          }
        }
        else{
          float r2 = (i*FD_cell_size_X)*(i*FD_cell_size_X) + (j*FD_cell_size_Y)*(j*FD_cell_size_Y) + (k*FD_cell_size_Z)*(k*FD_cell_size_Z);
          result += FD_cell_size_X * FD_cell_size_Y * FD_cell_size_Z *
                    ( (i*FD_cell_size_X) * __powf(r2,-1.5f) );
        }
      }
      result *= 1.0f/4.0f/3.14159265f*dim_inverse;

    }
  // ______________________________________________________________________________________________


  // for elements in Kernel component y ___________________________________________________________
    if (co==1){

      for(int cnta=-repetition_X; cnta<=repetition_X; cnta++)
      for(int cntb=-repetition_Y; cntb<=repetition_Y; cntb++)
      for(int cntc=-repetition_Z; cntc<=repetition_Z; cntc++){

        int i = a + cnta*Nkernel_X/2;
        int j = b + cntb*Nkernel_Y/2;
        int k = c + cntc*Nkernel_Z/2;
        int r2_int = i*i+j*j+k*k;

        if (r2_int<400){
          for (int=cnt1; cnt1<10; cnt1++){
            x = i*FD_cell_size_X + dev_qd_P_10_X[cnt1];
            for (int cnt2=0; cnt2<10; cnt2++){
              float y = j * FD_cell_size_Y + dev_qd_P_10_Y[cnt2];
              for (int cnt3=0; cnt3<10; cnt3++){
                float z = k * FD_cell_size_Z + dev_qd_P_10_Z[cnt3];
                result += FD_cell_size_X * FD_cell_size_Y * FD_cell_size_Z / 8.0f 
                          * dev_qd_W_10[cnt1] * dev_qd_W_10[cnt2] * dev_qd_W_10[cnt3]
                          * ( y*__powf(x*x+y*y+z*z, -1.5f) );
              }
            }
          }
        }
        else{
          float r2 = (i*FD_cell_size_X)*(i*FD_cell_size_X) + (j*FD_cell_size_Y)*(j*FD_cell_size_Y) + (k*FD_cell_size_Z)*(k*FD_cell_size_Z);
          result += FD_cell_size_X * FD_cell_size_Y * FD_cell_size_Z *
                    ( (i*FD_cell_size_X) * __powf(r2,-1.5f) );
        }
      }
      result *= 1.0f/4.0f/3.14159265f*dim_inverse;

    }
  // ______________________________________________________________________________________________

  // for elements in Kernel component z ___________________________________________________________
    if (co==2){

      for(int cnta=-repetition_X; cnta<=repetition_X; cnta++)
      for(int cntb=-repetition_Y; cntb<=repetition_Y; cntb++)
      for(int cntc=-repetition_Z; cntc<=repetition_Z; cntc++){

        int i = a + cnta*Nkernel_X/2;
        int j = b + cntb*Nkernel_Y/2;
        int k = c + cntc*Nkernel_Z/2;
        int r2_int = i*i+j*j+k*k;

        if (r2_int<400){
          for (int=cnt1; cnt1<10; cnt1++){
            x = i*FD_cell_size_X + dev_qd_P_10_X[cnt2];
            for (int cnt2=0; cnt2<10; cnt2++){
              float y = j * FD_cell_size_Y + dev_qd_P_10_Y[cnt2];
              for (int cnt3=0; cnt3<10; cnt3++){
                float z = k * FD_cell_size_Z + dev_qd_P_10_Z[cnt3];
                result += FD_cell_size_X * FD_cell_size_Y * FD_cell_size_Z / 8.0f 
                          * dev_qd_W_10[cnt1] * dev_qd_W_10[cnt2] * dev_qd_W_10[cnt3]
                          * ( z*__powf(x*x+y*y+z*z, -1.5f) );
              }
            }
          }
        }
        else{
          float r2 = (i*FD_cell_size_X)*(i*FD_cell_size_X) + (j*FD_cell_size_Y)*(j*FD_cell_size_Y) + (k*FD_cell_size_Z)*(k*FD_cell_size_Z);
          result += FD_cell_size_X * FD_cell_size_Y * FD_cell_size_Z *
                    ( (i*FD_cell_size_X) * __powf(r2,-1.5f) );
        }
      }
      result *= 1.0f/4.0f/3.14159265f*dim_inverse;

    }
  // ______________________________________________________________________________________________

  
  return( result );       //correct for scaling factor in FFTs
}


__global__ void _gpu_extract_real_parts_biot_savart3d(float *dev_kernel_array, float *dev_temp, int N){
  

  int e = threadindex;

  if (e<N)
    dev_kernel_array[e] = dev_temp[2*e];

  return;
}



void initialize_Gauss_quadrature_on_gpu_biot_savart3d(float *dev_qd_W_10, float *dev_qd_P_10, float *FD_cell_size){

  // initialize standard order 10 Gauss quadrature points and weights _____________________________
    float *std_qd_P_10 = (float*) calloc(10, sizeof(float));
    std_qd_P_10[0] = -0.97390652851717197f;
    std_qd_P_10[1] = -0.86506336668898498f;
    std_qd_P_10[2] = -0.67940956829902399f;
    std_qd_P_10[3] = -0.43339539412924699f;
    std_qd_P_10[4] = -0.14887433898163099f;
    std_qd_P_10[5] = -std_qd_P_10[4];
    std_qd_P_10[6] = -std_qd_P_10[3];
    std_qd_P_10[7] = -std_qd_P_10[2];
    std_qd_P_10[8] = -std_qd_P_10[1];
    std_qd_P_10[9] = -std_qd_P_10[0];
    float *host_qd_W_10 = (float*)calloc(10, sizeof(float));
    host_qd_W_10[0] = host_qd_W_10[9] = 0.066671344308687999f;
    host_qd_W_10[1] = host_qd_W_10[8] = 0.149451349150581f;
    host_qd_W_10[2] = host_qd_W_10[7] = 0.21908636251598201f;
    host_qd_W_10[3] = host_qd_W_10[6] = 0.26926671930999602f;
    host_qd_W_10[4] = host_qd_W_10[5] = 0.29552422471475298f;
  // ______________________________________________________________________________________________

  // Map the standard Gauss quadrature points to the used integration boundaries __________________
    float *host_qd_P_10 =  (float *) calloc (3*10, sizeof(float));
    get_Quad_Points_biot_savart3d(&host_qd_P_10[X*10], std_qd_P_10, 10, -0.5f*FD_cell_size[X], 0.5f*FD_cell_size[X]);
    get_Quad_Points_biot_savart3d(&host_qd_P_10[Y*10], std_qd_P_10, 10, -0.5f*FD_cell_size[Y], 0.5f*FD_cell_size[Y]);
    get_Quad_Points_biot_savart3d(&host_qd_P_10[Z*10], std_qd_P_10, 10, -0.5f*FD_cell_size[Z], 0.5f*FD_cell_size[Z]);
  // ______________________________________________________________________________________________

  // copy to the quadrature points and weights to the device ______________________________________
    memcpy_to_gpu (host_qd_W_10, dev_qd_W_10, 10);
    memcpy_to_gpu (host_qd_P_10, dev_qd_P_10, 3*10);
  // ______________________________________________________________________________________________

  free (std_qd_P_10);
  free (host_qd_P_10);
  free (host_qd_W_10);

  return;
}

void get_Quad_Points_biot_savart3d(float *gaussQP, float *stdGaussQP, int qOrder, double a, double b){

  int i;
  double A = (b-a)/2.0f; // coefficients for transformation x'= Ax+B
  double B = (a+b)/2.0f; // where x' is the new integration parameter
  
  for(i = 0; i < qOrder; i++)
    gaussQP[i] = A*stdGaussQP[i]+B;

  return;
}


#ifdef __cplusplus
}
#endif

