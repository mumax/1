#include "tensor.h"
#include "gputil.h"
#include "param.h"
//#include "gpufft2.h"
#include "gpu_fft6.h"
#include "gpu_fftbig.h"
#include "assert.h"
#include "timer.h"
#include <stdio.h>
#include "gpu_conf.h"
#include "gpu_micromag3d_kernel.h"

#ifdef __cplusplus
extern "C" {
#endif


tensor *gpu_micromag3d_kernel(param* p){
  
  // check input + allocate tensor on device ______________________________________________________
    check_param(p);
    int kernelStorageN = p->kernelSize[X] * p->kernelSize[Y] * (p->kernelSize[Z]+2);

    tensor *dev_kernel;
    if (p->size[X]==1)
      dev_kernel = as_tensor(new_gpu_array(4*kernelStorageN/2), 2, 4, kernelStorageN/2);  // only real parts!!
    else
      dev_kernel = as_tensor(new_gpu_array(6*kernelStorageN/2), 2, 6, kernelStorageN/2);  // only real parts!!
  // ______________________________________________________________________________________________


  // initialization Gauss quadrature points for integrations + copy to gpu ________________________
    float *dev_qd_W_10 = new_gpu_array(10);
    float *dev_qd_P_10 = new_gpu_array(3*10);
    initialize_Gauss_quadrature_on_gpu_micromag3d(dev_qd_W_10, dev_qd_P_10, p->cellSize);
  // ______________________________________________________________________________________________
  
  
  // Plan initialization of FFTs and initialization of the kernel _________________________________
    gpuFFT3dPlan* kernel_plan = new_gpuFFT3dPlan_padded(p->kernelSize, p->kernelSize);
    gpu_init_and_FFT_Greens_kernel_elements_micromag3d(dev_kernel->list, p->kernelSize, p->exchType, p->exchInConv, p->cellSize, p->demagPeriodic, dev_qd_P_10, dev_qd_W_10, kernel_plan);
  // ______________________________________________________________________________________________ 
  
  cudaFree (dev_qd_W_10);
  cudaFree (dev_qd_P_10);

  return (dev_kernel);
}



/// remark: number of FD cells in a dimension can not be odd if no zero padding!!
void gpu_init_and_FFT_Greens_kernel_elements_micromag3d(float *dev_kernel, int *kernelSize, int exchType, int *exchInConv, float *FD_cell_size, int *repetition, float *dev_qd_P_10, float *dev_qd_W_10, gpuFFT3dPlan* kernel_plan){

  
  int kernelN = kernelSize[X]*kernelSize[Y]*kernelSize[Z];                              // size of a kernel component without zeros
  float *dev_temp1 = new_gpu_array(kernelN);                                            // temp array on device for storage of kernel component without zeros (input of fft routine)
  int kernelStorageN = kernelSize[X]*kernelSize[Y]*(kernelSize[Z]+2);  // size of a zero padded kernel component
  float *dev_temp2 = new_gpu_array(kernelStorageN);                                     // temp array on device for storage of zero padded kernel component (output of fft routine)
  
  // Define gpugrids and blocks ___________________________________________________________________
    dim3 gridsize1((kernelSize[X]+1)/2,kernelSize[Y]/2, 1);
    dim3 blocksize1(kernelSize[Z]/2, 1,1);
    check3dconf(gridsize1, blocksize1);
    
    int N2 = kernelStorageN/2;
    dim3 gridsize2, blocksize2;
    make1dconf(N2, &gridsize2, &blocksize2);
  // ______________________________________________________________________________________________
  
  // Main function operations _____________________________________________________________________
    int rank0 = 0;                                      // defines the first rank of the Greens kernel, N0>1: [xx, xy, xz, yy, yz, zz], N1=1: [xx, yy, yz, zz]
    for (int co1=0; co1<3; co1++){                      // for a Greens kernel component [co1,co2]:
      for (int co2=co1; co2<3; co2++){
        if (co1==0 && co2>0 && kernelSize[0]==1)  continue;    // N0=1 -> xy and xz components have only zeros, so left out.
          // Put all elements in 'dev_temp1' to zero.
        gpu_zero(dev_temp1, kernelN);    
        gpu_sync();
          // Fill in the elements.
        _gpu_init_Greens_kernel_elements_micromag3d<<<gridsize1, blocksize1>>>(dev_temp1, kernelSize[X], kernelSize[Y], kernelSize[Z], exchType, exchInConv[X], exchInConv[Y], exchInConv[Z], co1, co2, FD_cell_size[X], FD_cell_size[Y], FD_cell_size[Z], repetition[X], repetition[Y], repetition[Z], dev_qd_P_10, dev_qd_W_10);
        gpu_sync();
          // Fourier transform the kernel component.
//         gpuFFT3dPlan_forward(kernel_plan, dev_temp1, dev_temp2); 
        gpuFFT3dPlan_forward(kernel_plan, dev_temp1, dev_temp2);
        gpu_sync();
          // Copy the real parts to the corresponding place in the dev_kernel tensor.
        _gpu_extract_real_parts_micromag3d<<<gridsize2, blocksize2>>>(&dev_kernel[rank0*kernelStorageN/2], dev_temp2, N2);
        gpu_sync();
        rank0++;                                        // get ready for next component
      }
    } 
  // ______________________________________________________________________________________________


  cudaFree (dev_temp1);
  cudaFree (dev_temp2);
  
  return;
}


__global__ void _gpu_init_Greens_kernel_elements_micromag3d(float *dev_temp, int Nkernel_X, int Nkernel_Y, int Nkernel_Z, int exchType, int exchInConv_X, int exchInConv_Y, int exchInConv_Z, int co1, int co2, float FD_cell_size_X, float FD_cell_size_Y, float FD_cell_size_Z, int repetition_X, int repetition_Y, int repetition_Z, float *dev_qd_P_10, float *dev_qd_W_10){

  /// @todo possible redeclaration of threadparameters required when using 'make3dconf' for thread launching.

  int i = blockIdx.x;
  int j = blockIdx.y;
  int k = threadIdx.x; 

  int N2 = Nkernel_Z;
  int N12 = Nkernel_Y * N2;

    dev_temp[            i*N12 +             j*N2 +           k] = _gpu_get_Greens_element_micromag3d(Nkernel_X, Nkernel_Y, Nkernel_Z, exchType, exchInConv_X, exchInConv_Y, exchInConv_Z, co1, co2,  i,  j,  k, FD_cell_size_X, FD_cell_size_Y, FD_cell_size_Z, repetition_X, repetition_Y, repetition_Z, dev_qd_P_10, dev_qd_W_10);
  if (i>0)
    dev_temp[(Nkernel_X-i)*N12 +             j*N2 +           k] = _gpu_get_Greens_element_micromag3d(Nkernel_X, Nkernel_Y, Nkernel_Z, exchType, exchInConv_X, exchInConv_Y, exchInConv_Z, co1, co2, -i,  j,  k, FD_cell_size_X, FD_cell_size_Y, FD_cell_size_Z, repetition_X, repetition_Y, repetition_Z, dev_qd_P_10, dev_qd_W_10);
  if (j>0)
    dev_temp[            i*N12 + (Nkernel_Y-j)*N2 +           k] = _gpu_get_Greens_element_micromag3d(Nkernel_X, Nkernel_Y, Nkernel_Z, exchType, exchInConv_X, exchInConv_Y, exchInConv_Z, co1, co2,  i, -j,  k, FD_cell_size_X, FD_cell_size_Y, FD_cell_size_Z, repetition_X, repetition_Y, repetition_Z, dev_qd_P_10, dev_qd_W_10);
  if (k>0) 
    dev_temp[            i*N12 +             j*N2 + Nkernel_Z-k] = _gpu_get_Greens_element_micromag3d(Nkernel_X, Nkernel_Y, Nkernel_Z, exchType, exchInConv_X, exchInConv_Y, exchInConv_Z, co1, co2,  i,  j, -k, FD_cell_size_X, FD_cell_size_Y, FD_cell_size_Z, repetition_X, repetition_Y, repetition_Z, dev_qd_P_10, dev_qd_W_10);
  if (i>0 && j>0)
    dev_temp[(Nkernel_X-i)*N12 + (Nkernel_Y-j)*N2 +           k] = _gpu_get_Greens_element_micromag3d(Nkernel_X, Nkernel_Y, Nkernel_Z, exchType, exchInConv_X, exchInConv_Y, exchInConv_Z, co1, co2, -i, -j,  k, FD_cell_size_X, FD_cell_size_Y, FD_cell_size_Z, repetition_X, repetition_Y, repetition_Z, dev_qd_P_10, dev_qd_W_10);
  if (i>0 && k>0) 
    dev_temp[(Nkernel_X-i)*N12 +             j*N2 + Nkernel_Z-k] = _gpu_get_Greens_element_micromag3d(Nkernel_X, Nkernel_Y, Nkernel_Z, exchType, exchInConv_X, exchInConv_Y, exchInConv_Z, co1, co2, -i,  j, -k, FD_cell_size_X, FD_cell_size_Y, FD_cell_size_Z, repetition_X, repetition_Y, repetition_Z, dev_qd_P_10, dev_qd_W_10);
  if (j>0 && k>0) 
    dev_temp[            i*N12 + (Nkernel_Y-j)*N2 + Nkernel_Z-k] = _gpu_get_Greens_element_micromag3d(Nkernel_X, Nkernel_Y, Nkernel_Z, exchType, exchInConv_X, exchInConv_Y, exchInConv_Z, co1, co2,  i, -j, -k, FD_cell_size_X, FD_cell_size_Y, FD_cell_size_Z, repetition_X, repetition_Y, repetition_Z, dev_qd_P_10, dev_qd_W_10);
  if (i>0 && j>0 && k>0) 
    dev_temp[(Nkernel_X-i)*N12 + (Nkernel_Y-j)*N2 + Nkernel_Z-k] = _gpu_get_Greens_element_micromag3d(Nkernel_X, Nkernel_Y, Nkernel_Z, exchType, exchInConv_X, exchInConv_Y, exchInConv_Z, co1, co2, -i, -j, -k, FD_cell_size_X, FD_cell_size_Y, FD_cell_size_Z, repetition_X, repetition_Y, repetition_Z, dev_qd_P_10, dev_qd_W_10);

  return;
}

__device__ float _gpu_get_Greens_element_micromag3d(int Nkernel_X, int Nkernel_Y, int Nkernel_Z, int exchType, int exchInConv_X, int exchInConv_Y, int exchInConv_Z, int co1, int co2, int a, int b, int c, float FD_cell_size_X, float FD_cell_size_Y, float FD_cell_size_Z, int repetition_X, int repetition_Y, int repetition_Z, float *dev_qd_P_10, float *dev_qd_W_10){

  float result = 0.0f;
  float *dev_qd_P_10_X = &dev_qd_P_10[X*10];
  float *dev_qd_P_10_Y = &dev_qd_P_10[Y*10];
  float *dev_qd_P_10_Z = &dev_qd_P_10[Z*10];
  float dim_inverse = 1.0f/( (float) Nkernel_X*Nkernel_Y*Nkernel_Z );
  
  // for elements in Kernel component gxx _________________________________________________________
    if (co1==0 && co2==0){

      for(int cnta=-repetition_X; cnta<=repetition_X; cnta++)
      for(int cntb=-repetition_Y; cntb<=repetition_Y; cntb++)
      for(int cntc=-repetition_Z; cntc<=repetition_Z; cntc++){

        int i = a + cnta*Nkernel_X/2;
        int j = b + cntb*Nkernel_Y/2;
        int k = c + cntc*Nkernel_Z/2;
        int r2_int = i*i+j*j+k*k;

        if (r2_int<400){
          float x1 = (i + 0.5f) * FD_cell_size_X;
          float x2 = (i - 0.5f) * FD_cell_size_X;
          for (int cnt2=0; cnt2<10; cnt2++){
            float y = j * FD_cell_size_Y + dev_qd_P_10_Y[cnt2];
            for (int cnt3=0; cnt3<10; cnt3++){
              float z = k * FD_cell_size_Z + dev_qd_P_10_Z[cnt3];
              result += FD_cell_size_Y * FD_cell_size_Z / 4.0f * dev_qd_W_10[cnt2] * dev_qd_W_10[cnt3] *
                ( x1*__powf(x1*x1+y*y+z*z, -1.5f) - x2*__powf(x2*x2+y*y+z*z, -1.5f));
            }
          }
        }
        else{
          float r2 = (i*FD_cell_size_X)*(i*FD_cell_size_X) + (j*FD_cell_size_Y)*(j*FD_cell_size_Y) + (k*FD_cell_size_Z)*(k*FD_cell_size_Z);
          result += FD_cell_size_X * FD_cell_size_Y * FD_cell_size_Z *
                    (1.0f/ __powf(r2,1.5f) - 3.0f* (i*FD_cell_size_X) * (i*FD_cell_size_X) * __powf(r2,-2.5f));
        }
      }
      result *= -1.0f/4.0f/3.14159265f;

      if (exchType==EXCH_6NGBR && exchInConv_X==1){
        if (a== 0 && b== 0 && c== 0 && Nkernel_X==1)  result -= 2.0f/FD_cell_size_Y/FD_cell_size_Y + 2.0f/FD_cell_size_Z/FD_cell_size_Z;
        if (a== 0 && b== 0 && c== 0 && Nkernel_X>1 )  result -= 2.0f/FD_cell_size_X/FD_cell_size_X + 2.0f/FD_cell_size_Y/FD_cell_size_Y + 2.0f/FD_cell_size_Z/FD_cell_size_Z;
        if (a== 1 && b== 0 && c== 0)  result += 1.0f/FD_cell_size_X/FD_cell_size_X;
        if (a==-1 && b== 0 && c== 0)  result += 1.0f/FD_cell_size_X/FD_cell_size_X;
        if (a== 0 && b== 1 && c== 0)  result += 1.0f/FD_cell_size_Y/FD_cell_size_Y;
        if (a== 0 && b==-1 && c== 0)  result += 1.0f/FD_cell_size_Y/FD_cell_size_Y;
        if (a== 0 && b== 0 && c== 1)  result += 1.0f/FD_cell_size_Z/FD_cell_size_Z;
        if (a== 0 && b== 0 && c==-1)  result += 1.0f/FD_cell_size_Z/FD_cell_size_Z;
      }
    }
  // ______________________________________________________________________________________________


  // for elements in Kernel component gxy _________________________________________________________
    if (co1==0 && co2==1){
      for(int cnta=-repetition_X; cnta<=repetition_X; cnta++)
      for(int cntb=-repetition_Y; cntb<=repetition_Y; cntb++)
      for(int cntc=-repetition_Z; cntc<=repetition_Z; cntc++){

        int i = a + cnta*Nkernel_X/2;
        int j = b + cntb*Nkernel_Y/2;
        int k = c + cntc*Nkernel_Z/2;
        int r2_int = i*i+j*j+k*k;

        if (r2_int<400){
          float x1 = (i + 0.5f) * FD_cell_size_X;
          float x2 = (i - 0.5f) * FD_cell_size_X;
          for (int cnt2=0; cnt2<10; cnt2++){
            float y = j * FD_cell_size_Y + dev_qd_P_10_Y[cnt2];
            for (int cnt3=0; cnt3<10; cnt3++){
              float z = k * FD_cell_size_Z + dev_qd_P_10_Z[cnt3];
              result += FD_cell_size_Y * FD_cell_size_Z / 4.0f * dev_qd_W_10[cnt2] * dev_qd_W_10[cnt3] *
                ( y*__powf(x1*x1+y*y+z*z, -1.5f) - y*__powf(x2*x2+y*y+z*z, -1.5f));
            }
          }
        }
        else{
          float r2 = (i*FD_cell_size_X)*(i*FD_cell_size_X) + (j*FD_cell_size_Y)*(j*FD_cell_size_Y) + (k*FD_cell_size_Z)*(k*FD_cell_size_Z);
          result += FD_cell_size_X * FD_cell_size_Y * FD_cell_size_Z * 
                    (- 3.0f* (i*FD_cell_size_X) * (j*FD_cell_size_Y) * __powf(r2,-2.5f));
        }
      }
      result *= -1.0f/4.0f/3.14159265f;
    }
  // ______________________________________________________________________________________________


  // for elements in Kernel component gxz _________________________________________________________
    if (co1==0 && co2==2){
      for(int cnta=-repetition_X; cnta<=repetition_X; cnta++)
      for(int cntb=-repetition_Y; cntb<=repetition_Y; cntb++)
      for(int cntc=-repetition_Z; cntc<=repetition_Z; cntc++){

        int i = a + cnta*Nkernel_X/2;
        int j = b + cntb*Nkernel_Y/2;
        int k = c + cntc*Nkernel_Z/2;
        int r2_int = i*i+j*j+k*k;

        if (r2_int<400){
          float x1 = (i + 0.5f) * FD_cell_size_X;
          float x2 = (i - 0.5f) * FD_cell_size_X;
          for (int cnt2=0; cnt2<10; cnt2++){
            float y = j * FD_cell_size_Y + dev_qd_P_10_Y[cnt2];
            for (int cnt3=0; cnt3<10; cnt3++){
              float z = k * FD_cell_size_Z + dev_qd_P_10_Z[cnt3];
              result += FD_cell_size_Y * FD_cell_size_Z / 4.0f * dev_qd_W_10[cnt2] * dev_qd_W_10[cnt3] *
                ( z*__powf(x1*x1+y*y+z*z, -1.5f) - z*__powf(x2*x2+y*y+z*z, -1.5f));
            }
          }
        }
        else{
          float r2 = (i*FD_cell_size_X)*(i*FD_cell_size_X) + (j*FD_cell_size_Y)*(j*FD_cell_size_Y) + (k*FD_cell_size_Z)*(k*FD_cell_size_Z);
          result += FD_cell_size_X * FD_cell_size_Y * FD_cell_size_Z * 
                    (- 3.0f* (i*FD_cell_size_X) * (k*FD_cell_size_Y) * __powf(r2,-2.5f));
        }
      }
      result *= -1.0f/4.0f/3.14159265f;
    }
  // ______________________________________________________________________________________________


  // for elements in Kernel component gyy _________________________________________________________
    if (co1==1 && co2==1){
      for(int cnta=-repetition_X; cnta<=repetition_X; cnta++)
      for(int cntb=-repetition_Y; cntb<=repetition_Y; cntb++)
      for(int cntc=-repetition_Z; cntc<=repetition_Z; cntc++){

        int i = a + cnta*Nkernel_X/2;
        int j = b + cntb*Nkernel_Y/2;
        int k = c + cntc*Nkernel_Z/2;
        int r2_int = i*i+j*j+k*k;

        if (r2_int<400){
          float y1 = (j + 0.5f) * FD_cell_size_Y;
          float y2 = (j - 0.5f) * FD_cell_size_Y;
          for (int cnt1=0; cnt1<10; cnt1++){
            float x = i * FD_cell_size_X + dev_qd_P_10_X[cnt1];
            for (int cnt3=0; cnt3<10; cnt3++){
              float z = k * FD_cell_size_Z + dev_qd_P_10_Z[cnt3];
              result += FD_cell_size_X * FD_cell_size_Z / 4.0f * dev_qd_W_10[cnt1] * dev_qd_W_10[cnt3] *
                ( y1*__powf(x*x+y1*y1+z*z, -1.5f) - y2*__powf(x*x+y2*y2+z*z, -1.5f));
            }
          }
        }
        else{
          float r2 = (i*FD_cell_size_X)*(i*FD_cell_size_X) + (j*FD_cell_size_Y)*(j*FD_cell_size_Y) + (k*FD_cell_size_Z)*(k*FD_cell_size_Z);
          result += FD_cell_size_X * FD_cell_size_Y * FD_cell_size_Z * 
                    (1.0f/ __powf(r2,1.5f) - 3.0f* (j*FD_cell_size_Y) * (j*FD_cell_size_Y) * __powf(r2,-2.5f));
        }
      }
      result *= -1.0f/4.0f/3.14159265f;

      if (exchType==EXCH_6NGBR && exchInConv_Y==1){
        if (a== 0 && b== 0 && c== 0 && Nkernel_X==1)  result -= 2.0f/FD_cell_size_Y/FD_cell_size_Y + 2.0f/FD_cell_size_Z/FD_cell_size_Z;
        if (a== 0 && b== 0 && c== 0 && Nkernel_X>1 )  result -= 2.0f/FD_cell_size_X/FD_cell_size_X + 2.0f/FD_cell_size_Y/FD_cell_size_Y + 2.0f/FD_cell_size_Z/FD_cell_size_Z;
        if (a== 1 && b== 0 && c== 0)  result += 1.0f/FD_cell_size_X/FD_cell_size_X;
        if (a==-1 && b== 0 && c== 0)  result += 1.0f/FD_cell_size_X/FD_cell_size_X;
        if (a== 0 && b== 1 && c== 0)  result += 1.0f/FD_cell_size_Y/FD_cell_size_Y;
        if (a== 0 && b==-1 && c== 0)  result += 1.0f/FD_cell_size_Y/FD_cell_size_Y;
        if (a== 0 && b== 0 && c== 1)  result += 1.0f/FD_cell_size_Z/FD_cell_size_Z;
        if (a== 0 && b== 0 && c==-1)  result += 1.0f/FD_cell_size_Z/FD_cell_size_Z;
      }
    }
// ______________________________________________________________________________________________


  // for elements in Kernel component gyz _________________________________________________________
    if (co1==1 && co2==2){
      for(int cnta=-repetition_X; cnta<=repetition_X; cnta++)
      for(int cntb=-repetition_Y; cntb<=repetition_Y; cntb++)
      for(int cntc=-repetition_Z; cntc<=repetition_Z; cntc++){

        int i = a + cnta*Nkernel_X/2;
        int j = b + cntb*Nkernel_Y/2;
        int k = c + cntc*Nkernel_Z/2;
        int r2_int = i*i+j*j+k*k;

        if (r2_int<400){
          float y1 = (j + 0.5f) * FD_cell_size_Y;
          float y2 = (j - 0.5f) * FD_cell_size_Y;
          for (int cnt1=0; cnt1<10; cnt1++){
            float x = i * FD_cell_size_X + dev_qd_P_10_X[cnt1];
            for (int cnt3=0; cnt3<10; cnt3++){
              float z = k * FD_cell_size_Z + dev_qd_P_10_Z[cnt3];
              result += FD_cell_size_X * FD_cell_size_Z / 4.0f * dev_qd_W_10[cnt1] * dev_qd_W_10[cnt3] *
                ( z*__powf(x*x+y1*y1+z*z, -1.5f) - z*__powf(x*x+y2*y2+z*z, -1.5f));
            }
          }
        }
        else{
          float r2 = (i*FD_cell_size_X)*(i*FD_cell_size_X) + (j*FD_cell_size_Y)*(j*FD_cell_size_Y) + (k*FD_cell_size_Z)*(k*FD_cell_size_Z);
          result += FD_cell_size_X * FD_cell_size_Y * FD_cell_size_Z * 
                    ( - 3.0f* (j*FD_cell_size_Y) * (k*FD_cell_size_Z) * __powf(r2,-2.5f));
        }
      }
      result *= -1.0f/4.0f/3.14159265f;
    }
  // ______________________________________________________________________________________________


  // for elements in Kernel component gzz _________________________________________________________
    if (co1==2 && co2==2){
      for(int cnta=-repetition_X; cnta<=repetition_X; cnta++)
      for(int cntb=-repetition_Y; cntb<=repetition_Y; cntb++)
      for(int cntc=-repetition_Z; cntc<=repetition_Z; cntc++){

        int i = a + cnta*Nkernel_X/2;
        int j = b + cntb*Nkernel_Y/2;
        int k = c + cntc*Nkernel_Z/2;
        int r2_int = i*i+j*j+k*k;

        if (r2_int<400){
          float z1 = (k + 0.5f) * FD_cell_size_Z;
          float z2 = (k - 0.5f) * FD_cell_size_Z;
          for (int cnt1=0; cnt1<10; cnt1++){
            float x = i * FD_cell_size_X + dev_qd_P_10_X[cnt1];
            for (int cnt2=0; cnt2<10; cnt2++){
              float y = j * FD_cell_size_Y + dev_qd_P_10_Y[cnt2];
              result += FD_cell_size_X * FD_cell_size_Y / 4.0f * dev_qd_W_10[cnt1] * dev_qd_W_10[cnt2] *
                ( z1*__powf(x*x+y*y+z1*z1, -1.5f) - z2*__powf(x*x+y*y+z2*z2, -1.5f));
            }
          }
        }
        else{
          float r2 = (i*FD_cell_size_X)*(i*FD_cell_size_X) + (j*FD_cell_size_Y)*(j*FD_cell_size_Y) + (k*FD_cell_size_Z)*(k*FD_cell_size_Z);
          result += FD_cell_size_X * FD_cell_size_Y * FD_cell_size_Z * 
                    (1.0f/ __powf(r2,1.5f) - 3.0f* (k*FD_cell_size_Z) * (k*FD_cell_size_Z) * __powf(r2,-2.5f));
        }
      }
      result *= -1.0f/4.0f/3.14159265f;

      if (exchType==EXCH_6NGBR && exchInConv_Z==1){
        if (a== 0 && b== 0 && c== 0 && Nkernel_X==1)  result -= 2.0f/FD_cell_size_Y/FD_cell_size_Y + 2.0f/FD_cell_size_Z/FD_cell_size_Z;
        if (a== 0 && b== 0 && c== 0 && Nkernel_X>1 )  result -= 2.0f/FD_cell_size_X/FD_cell_size_X + 2.0f/FD_cell_size_Y/FD_cell_size_Y + 2.0f/FD_cell_size_Z/FD_cell_size_Z;
        if (a== 1 && b== 0 && c== 0)  result += 1.0f/FD_cell_size_X/FD_cell_size_X;
        if (a==-1 && b== 0 && c== 0)  result += 1.0f/FD_cell_size_X/FD_cell_size_X;
        if (a== 0 && b== 1 && c== 0)  result += 1.0f/FD_cell_size_Y/FD_cell_size_Y;
        if (a== 0 && b==-1 && c== 0)  result += 1.0f/FD_cell_size_Y/FD_cell_size_Y;
        if (a== 0 && b== 0 && c== 1)  result += 1.0f/FD_cell_size_Z/FD_cell_size_Z;
        if (a== 0 && b== 0 && c==-1)  result += 1.0f/FD_cell_size_Z/FD_cell_size_Z;
      }
    }
  // ______________________________________________________________________________________________
  
  return( result*dim_inverse );       //correct for scaling factor in FFTs
}

__global__ void _gpu_extract_real_parts_micromag3d(float *dev_kernel_array, float *dev_temp, int N){
  

  int e = threadindex;

  if (e<N)
    dev_kernel_array[e] = dev_temp[2*e];

  return;
}



void initialize_Gauss_quadrature_on_gpu_micromag3d(float *dev_qd_W_10, float *dev_qd_P_10, float *FD_cell_size){

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
    get_Quad_Points_micromag3d(&host_qd_P_10[X*10], std_qd_P_10, 10, -0.5f*FD_cell_size[X], 0.5f*FD_cell_size[X]);
    get_Quad_Points_micromag3d(&host_qd_P_10[Y*10], std_qd_P_10, 10, -0.5f*FD_cell_size[Y], 0.5f*FD_cell_size[Y]);
    get_Quad_Points_micromag3d(&host_qd_P_10[Z*10], std_qd_P_10, 10, -0.5f*FD_cell_size[Z], 0.5f*FD_cell_size[Z]);
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

void get_Quad_Points_micromag3d(float *gaussQP, float *stdGaussQP, int qOrder, double a, double b){

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



// remove the following if code contains no errors for sure.

//   float *host_temp = (float *)calloc(kernelStorageN, sizeof(float));      // temp array on host for storage of each component in real + i*complex format in serie (only for debugging purposes)
//   float *host_temp2 = (float *)calloc(kernelStorageN/2, sizeof(float));   // temp array on host for storage of only the real components
// 
//   int testco1 = 0;
//   int testco2 = 0;
//   int testrang = 0;
//   for (int i=0; i<testco1; i++)
//     for (int j=i; j<testco2; j++)
//       testrang ++;
//   fprintf(stderr, "test co: %d, %d, testrang: %d\n\n", testco1, testco2, testrang);
// 
//   gpu_zero(dev_temp, kernelStorageN);
//   gpu_sync();
// //  _gpu_init_Greens_kernel_elements<<<gridsize1, blocksize1>>>(dev_temp, Nkernel[X], Nkernel[Y], Nkernel[Z], testco1, testco2, FD_cell_size[X], FD_cell_size[Y], FD_cell_size[Z], cst, repetition[X], repetition[Y], repetition[Z], dev_qd_P_10, dev_qd_W_10);
//   _gpu_init_Greens_kernel_elements_micromag3d<<<gridsize1, blocksize1>>>(dev_temp, kernelSize[X], kernelSize[Y], kernelSize[Z], kernelStorageSize[Z], exchInConv[X], exchInConv[Y], exchInConv[Z], testco1, testco2, FD_cell_size[X], FD_cell_size[Y], FD_cell_size[Z], repetition[X], repetition[Y], repetition[Z], dev_qd_P_10, dev_qd_W_10);
//   gpu_sync();
// 
//   memcpy_from_gpu(dev_temp, host_temp, kernelStorageN);
//   gpu_sync();
//   fprintf(stderr, "\nkernel elements (untransformed), co: %d, %d:\n", testco1, testco2);
//   for (int i=0; i<kernelStorageSize[X]; i++){
//     for (int j=0; j<kernelStorageSize[Y]; j++){
//       for (int k=0; k<kernelStorageSize[Z]; k++){
//         fprintf(stderr, "%e ", host_temp[i*kernelStorageSize[Y]*kernelStorageSize[Z] + j*kernelStorageSize[Z] + k]);
//       }
//       fprintf(stderr, "\n");
//     }
//     fprintf(stderr, "\n");
//   }
//   
// 
//   gpuFFT3dPlan_forward(kernel_plan, FFT_input, FFT_output); 
//   gpu_sync();
//   
//   memcpy_from_gpu(dev_temp, host_temp, kernelStorageN);
//   gpu_sync();
//   fprintf(stderr, "\nkernel elements (transformed), co: %d, %d:\n", testco1, testco2);
//   for (int i=0; i<kernelStorageSize[X]; i++){
//     for (int j=0; j<kernelStorageSize[Y]; j++){
//       for (int k=0; k<kernelStorageSize[Z]; k++){
//         fprintf(stderr, "%e ", host_temp[i*kernelStorageSize[Y]*kernelStorageSize[Z] + j*kernelStorageSize[Z] + k]);
//       }
//       fprintf(stderr, "\n");
//     }
//     fprintf(stderr, "\n");
//   }
// 
//   _gpu_extract_real_parts_micromag3d<<<gridsize2, blocksize2>>>(&dev_kernel->list[testrang*kernelStorageN/2], dev_temp);
//   gpu_sync();
//   fprintf(stderr, "\nkernel elements (transformed, real parts), co: %d, %d:\n", testco1, testco2);
//   memcpy_from_gpu(&dev_kernel->list[testrang*kernelStorageN/2], host_temp2, kernelStorageN/2);
//   gpu_sync();
// 
//   for (int i=0; i<kernelStorageSize[X]; i++){
//     for (int j=0; j<kernelStorageSize[Y]; j++){
//       for (int k=0; k<kernelStorageSize[Z]/2; k++){
//         fprintf(stderr, "%e ", host_temp2[i*kernelStorageSize[Y]*kernelStorageSize[Z]/2 + j*kernelStorageSize[Z]/2 + k]);
//       }
//       fprintf(stderr, "\n");
//     }
//     fprintf(stderr, "\n");
//   }
// 
