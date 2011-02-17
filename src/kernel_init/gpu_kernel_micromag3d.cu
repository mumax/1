#include "tensor.h"
#include "gputil.h"
#include "gpu_fft6.h"
#include "gpu_fftbig.h"
#include "assert.h"
#include "timer.h"
#include <stdio.h>
#include "gpu_conf.h"
#include "gpu_kernel_micromag3d.h"

#ifdef __cplusplus
extern "C" {
#endif

#define BLOCKSIZE 16


// tensor *gpu_micromag3d_kernel(int *kernelSize, float *cellsize, int exchType, int *exchInConv, int *repetition){
void gpu_kernel_micromag3d(int *kernelSize, float *cellsize, int exchType, int *exchInConv, int *repetition){
  
  // check input + allocate tensor on device ______________________________________________________
    int kernelStorageN = kernelSize[X] * kernelSize[Y] * (kernelSize[Z]+2);

    tensor *dev_kernel;
    if (p->size[X]==1)
      dev_kernel = as_tensor(new_gpu_array(4*kernelStorageN/2), 2, 4, kernelStorageN/2);  // only real parts!!
    else
      dev_kernel = as_tensor(new_gpu_array(6*kernelStorageN/2), 2, 6, kernelStorageN/2);  // only real parts!!
  // ______________________________________________________________________________________________


  // initialization Gauss quadrature points for integrations + copy to gpu ________________________
    float *dev_qd_W_10 = new_gpu_array(10);
    float *dev_qd_P_10 = new_gpu_array(3*10);
    initialize_Gauss_quadrature_on_gpu_micromag3d(dev_qd_W_10, dev_qd_P_10, cellSize);
  // ______________________________________________________________________________________________
  
  
  // Plan initialization of FFTs and initialization of the kernel _________________________________
    gpuFFT3dPlan* kernel_plan = new_gpuFFT3dPlan_padded(kernelSize, kernelSize);
    gpu_init_and_FFT_Greens_kernel_elements_micromag3d(dev_kernel->list, kernelSize, exchType, exchInConv, cellSize, repetition, dev_qd_P_10, dev_qd_W_10, kernel_plan);
  // ______________________________________________________________________________________________ 
  
  delete_FFT3dPlan(kernel_plan);
  cudaFree (dev_qd_W_10);
  cudaFree (dev_qd_P_10);

  write_tensor(dev_kernel, stdout);

  return;

}



/// remark: number of FD cells in a dimension can not be odd if no zero padding!!
void gpu_init_and_FFT_Greens_kernel_elements_micromag3d(float *dev_kernel, int *kernelSize, int exchType, int *exchInConv, float *FD_cell_size, int *repetition, float *dev_qd_P_10, float *dev_qd_W_10, gpuFFT3dPlan* kernel_plan){

  
  int kernelN = kernelSize[X]*kernelSize[Y]*kernelSize[Z];                              // size of a kernel component without zeros
  float *dev_temp1 = new_gpu_array(kernelN);                                            // temp array on device for storage of kernel component without zeros (input of fft routine)
  int kernelStorageN = kernelSize[X]*kernelSize[Y]*(kernelSize[Z]+2);  // size of a zero padded kernel component
  float *dev_temp2 = new_gpu_array(kernelStorageN);                                     // temp array on device for storage of zero padded kernel component (output of fft routine)
  
  // Define gpugrids and blocks ___________________________________________________________________
//     dim3 gridsize1((kernelSize[X]+1)/2,kernelSize[Y]/2, 1);
//     dim3 blocksize1(kernelSize[Z]/2, 1,1);
//     check3dconf(gridsize1, blocksize1);

    dim3 gridsize1((kernelSize[X]+1)/2,kernelSize[Y]/2, 1);
    dim3 blocksize1(1,1,1);
    check3dconf(gridsize1, blocksize1);

//     dim3 gridsize1(divUp((kernelSize[X]+1)/2, BLOCKSIZE), divUp(kernelSize[Y]/2, BLOCKSIZE), 1);
//     dim3 blocksize1(BLOCKSIZE, BLOCKSIZE,1);
//     check3dconf(gridsize1, blocksize1);

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
        _gpu_init_Greens_kernel_elements_micromag3d<<<gridsize1, blocksize1>>>(dev_temp1, kernelSize[X], kernelSize[Y], kernelSize[Z], exchType, exchInConv[X], exchInConv[Y], exchInConv[Z], co1, co2, FD_cell_size_X, FD_cell_size_Y, FD_cell_size_Z, repetition[X], repetition[Y], repetition[Z], dev_qd_P_10, dev_qd_W_10);
        gpu_sync();
          // Fourier transform the kernel component.
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

/*  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;*/
  int i = blockIdx.x;
  int j = blockIdx.y;
//   int k = threadIdx.x; 

  int N2 = Nkernel_Z;
  int N12 = Nkernel_Y * N2;

  if ( i<((Nkernel_X+1)/2) && j<(Nkernel_Y/2) )
    for (int k=0; k<N2/2; k++){
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
    }
  
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
        
/*        if (r2_int<400){   // The source is considered as a uniformly magnetized FD cell and the field is averaged over the complete observation FD cell
          for (int cnta=0; cnta<10; cnta++)
          for (int cntb=0; cntb<10; cntb++)
          for (int cntc=0; cntc<10; cntc++){
            float x1 = (i + 0.5f) * FD_cell_size_X + dev_qd_P_10_X[cnta];
            float x2 = (i - 0.5f) * FD_cell_size_X + dev_qd_P_10_X[cnta];
            for (int cnt2=0; cnt2<10; cnt2++){
              float y = j * FD_cell_size_Y + dev_qd_P_10_Y[cnt2] + dev_qd_P_10_Y[cntb];
              for (int cnt3=0; cnt3<10; cnt3++){
                float z = k * FD_cell_size_Z + dev_qd_P_10_Z[cnt3] + dev_qd_P_10_Z[cntc];
                result += 1.0f/8.0f * FD_cell_size_Y * FD_cell_size_Z / 4.0f * dev_qd_W_10[cnt2] * dev_qd_W_10[cnt3] *
                  ( x1*__powf(x1*x1+y*y+z*z, -1.5f) - x2*__powf(x2*x2+y*y+z*z, -1.5f));
              }
            }
          }
        }
        else{   // The source is considered as a point. No averaging of the field in the observation FD cell
          float r2 = (i*FD_cell_size_X)*(i*FD_cell_size_X) + (j*FD_cell_size_Y)*(j*FD_cell_size_Y) + (k*FD_cell_size_Z)*(k*FD_cell_size_Z);
          result += FD_cell_size_X * FD_cell_size_Y * FD_cell_size_Z *
                    (1.0f/ __powf(r2,1.5f) - 3.0f* (i*FD_cell_size_X) * (i*FD_cell_size_X) * __powf(r2,-2.5f));
        }*/
        
      }
      result *= -1.0f/4.0f/3.14159265f*dim_inverse;

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
      if (exchType==EXCH_12NGBR && exchInConv_X==1){
        if (a== 0 && b== 0 && c== 0 && Nkernel_X==1)  result -= 5.0f/2.0f/FD_cell_size_Y/FD_cell_size_Y + 5.0f/2.0f/FD_cell_size_Z/FD_cell_size_Z;
        if (a== 0 && b== 0 && c== 0 && Nkernel_X>1 )  result -= 5.0f/2.0f/FD_cell_size_X/FD_cell_size_X + 5.0f/2.0f/FD_cell_size_Y/FD_cell_size_Y + 5.0f/2.0f/FD_cell_size_Z/FD_cell_size_Z;
        if (a== 1 && b== 0 && c== 0)  result += 4.0f/3.0f/D_cell_size[X]/FD_cell_size_X;
        if (a==-1 && b== 0 && c== 0)  result += 4.0f/3.0f/FD_cell_size_X/FD_cell_size_X;
        if (a== 0 && b== 1 && c== 0)  result += 4.0f/3.0f/FD_cell_size_Y/FD_cell_size_Y;
        if (a== 0 && b==-1 && c== 0)  result += 4.0f/3.0f/FD_cell_size_Y/FD_cell_size_Y;
        if (a== 0 && b== 0 && c== 1)  result += 4.0f/3.0f/FD_cell_size_Z/FD_cell_size_Z;
        if (a== 0 && b== 0 && c==-1)  result += 4.0f/3.0f/FD_cell_size_Z/FD_cell_size_Z;
        if (a== 2 && b== 0 && c== 0)  result -= 1.0f/12.0f/D_cell_size[X]/FD_cell_size_X;
        if (a==-2 && b== 0 && c== 0)  result -= 1.0f/12.0f/FD_cell_size_X/FD_cell_size_X;
        if (a== 0 && b== 2 && c== 0)  result -= 1.0f/12.0f/FD_cell_size_Y/FD_cell_size_Y;
        if (a== 0 && b==-2 && c== 0)  result -= 1.0f/12.0f/FD_cell_size_Y/FD_cell_size_Y;
        if (a== 0 && b== 0 && c== 2)  result -= 1.0f/12.0f/FD_cell_size_Z/FD_cell_size_Z;
        if (a== 0 && b== 0 && c==-2)  result -= 1.0f/12.0f/FD_cell_size_Z/FD_cell_size_Z;
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

//         if (r2_int<400){
//           for (int cnta=0; cnta<10; cnta++)
//           for (int cntb=0; cntb<10; cntb++)
//           for (int cntc=0; cntc<10; cntc++){
//             float x1 = (i + 0.5f) * FD_cell_size_X + dev_qd_P_10_X[cnta];
//             float x2 = (i - 0.5f) * FD_cell_size_X + dev_qd_P_10_X[cnta];
//             for (int cnt2=0; cnt2<10; cnt2++){
//               float y = j * FD_cell_size_Y + dev_qd_P_10_Y[cnt2] + dev_qd_P_10_Y[cntb];
//               for (int cnt3=0; cnt3<10; cnt3++){
//                 float z = k * FD_cell_size_Z + dev_qd_P_10_Z[cnt3] + dev_qd_P_10_Z[cntc];
//                 result += 1.0f/8.0f * FD_cell_size_Y * FD_cell_size_Z / 4.0f * dev_qd_W_10[cnt2] * dev_qd_W_10[cnt3] *
//                   ( y*__powf(x1*x1+y*y+z*z, -1.5f) - y*__powf(x2*x2+y*y+z*z, -1.5f));
//               }
//             }
//           }
//         }
//         else{
//           float r2 = (i*FD_cell_size_X)*(i*FD_cell_size_X) + (j*FD_cell_size_Y)*(j*FD_cell_size_Y) + (k*FD_cell_size_Z)*(k*FD_cell_size_Z);
//           result += FD_cell_size_X * FD_cell_size_Y * FD_cell_size_Z * 
//                     (- 3.0f* (i*FD_cell_size_X) * (j*FD_cell_size_Y) * __powf(r2,-2.5f));
//         }
        
        
      }
      result *= -1.0f/4.0f/3.14159265f*dim_inverse;
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
        
/*        if (r2_int<400){
          for (int cnta=0; cnta<10; cnta++)
          for (int cntb=0; cntb<10; cntb++)
          for (int cntc=0; cntc<10; cntc++){
            float x1 = (i + 0.5f) * FD_cell_size_X + dev_qd_P_10_X[cnta];
            float x2 = (i - 0.5f) * FD_cell_size_X + dev_qd_P_10_X[cnta];
            for (int cnt2=0; cnt2<10; cnt2++){
              float y = j * FD_cell_size_Y + dev_qd_P_10_Y[cnt2] + dev_qd_P_10_Y[cntb];
              for (int cnt3=0; cnt3<10; cnt3++){
                float z = k * FD_cell_size_Z + dev_qd_P_10_Z[cnt3] + dev_qd_P_10_Z[cntc];
                result += 1.0f/8.0f * FD_cell_size_Y * FD_cell_size_Z / 4.0f * dev_qd_W_10[cnt2] * dev_qd_W_10[cnt3] *
                  ( z*__powf(x1*x1+y*y+z*z, -1.5f) - z*__powf(x2*x2+y*y+z*z, -1.5f));
              }
            }
          }
        }
        else{
          float r2 = (i*FD_cell_size_X)*(i*FD_cell_size_X) + (j*FD_cell_size_Y)*(j*FD_cell_size_Y) + (k*FD_cell_size_Z)*(k*FD_cell_size_Z);
          result += FD_cell_size_X * FD_cell_size_Y * FD_cell_size_Z * 
                    (- 3.0f* (i*FD_cell_size_X) * (k*FD_cell_size_Y) * __powf(r2,-2.5f));
        }*/
        
      }
      result *= -1.0f/4.0f/3.14159265f*dim_inverse;
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
        
/*        if (r2_int<400){
          for (int cnta=0; cnta<10; cnta++)
          for (int cntb=0; cntb<10; cntb++)
          for (int cntc=0; cntc<10; cntc++){
            float y1 = (j + 0.5f) * FD_cell_size_Y + dev_qd_P_10_Y[cntb];
            float y2 = (j - 0.5f) * FD_cell_size_Y + dev_qd_P_10_Y[cntb];
            for (int cnt1=0; cnt1<10; cnt1++){
              float x = i * FD_cell_size_X + dev_qd_P_10_X[cnt1] + dev_qd_P_10_X[cnta];
              for (int cnt3=0; cnt3<10; cnt3++){
                float z = k * FD_cell_size_Z + dev_qd_P_10_Z[cnt3] + dev_qd_P_10_Z[cntc];
                result += 1.0f/8.0f * FD_cell_size_X * FD_cell_size_Z / 4.0f * dev_qd_W_10[cnt1] * dev_qd_W_10[cnt3] *
                  ( y1*__powf(x*x+y1*y1+z*z, -1.5f) - y2*__powf(x*x+y2*y2+z*z, -1.5f));
              }
            }
          }
        }
        else{
          float r2 = (i*FD_cell_size_X)*(i*FD_cell_size_X) + (j*FD_cell_size_Y)*(j*FD_cell_size_Y) + (k*FD_cell_size_Z)*(k*FD_cell_size_Z);
          result += FD_cell_size_X * FD_cell_size_Y * FD_cell_size_Z * 
                    (1.0f/ __powf(r2,1.5f) - 3.0f* (j*FD_cell_size_Y) * (j*FD_cell_size_Y) * __powf(r2,-2.5f));
        }*/
        
        
      }
      result *= -1.0f/4.0f/3.14159265f*dim_inverse;

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
      if (exchType==EXCH_12NGBR && exchInConv_Y==1){
        if (a== 0 && b== 0 && c== 0 && Nkernel_X==1)  result -= 5.0f/2.0f/FD_cell_size_Y/FD_cell_size_Y + 5.0f/2.0f/FD_cell_size_Z/FD_cell_size_Z;
        if (a== 0 && b== 0 && c== 0 && Nkernel_X>1 )  result -= 5.0f/2.0f/FD_cell_size_X/FD_cell_size_X + 5.0f/2.0f/FD_cell_size_Y/FD_cell_size_Y + 5.0f/2.0f/FD_cell_size_Z/FD_cell_size_Z;
        if (a== 1 && b== 0 && c== 0)  result += 4.0f/3.0f/D_cell_size[X]/FD_cell_size_X;
        if (a==-1 && b== 0 && c== 0)  result += 4.0f/3.0f/FD_cell_size_X/FD_cell_size_X;
        if (a== 0 && b== 1 && c== 0)  result += 4.0f/3.0f/FD_cell_size_Y/FD_cell_size_Y;
        if (a== 0 && b==-1 && c== 0)  result += 4.0f/3.0f/FD_cell_size_Y/FD_cell_size_Y;
        if (a== 0 && b== 0 && c== 1)  result += 4.0f/3.0f/FD_cell_size_Z/FD_cell_size_Z;
        if (a== 0 && b== 0 && c==-1)  result += 4.0f/3.0f/FD_cell_size_Z/FD_cell_size_Z;
        if (a== 2 && b== 0 && c== 0)  result -= 1.0f/12.0f/D_cell_size[X]/FD_cell_size_X;
        if (a==-2 && b== 0 && c== 0)  result -= 1.0f/12.0f/FD_cell_size_X/FD_cell_size_X;
        if (a== 0 && b== 2 && c== 0)  result -= 1.0f/12.0f/FD_cell_size_Y/FD_cell_size_Y;
        if (a== 0 && b==-2 && c== 0)  result -= 1.0f/12.0f/FD_cell_size_Y/FD_cell_size_Y;
        if (a== 0 && b== 0 && c== 2)  result -= 1.0f/12.0f/FD_cell_size_Z/FD_cell_size_Z;
        if (a== 0 && b== 0 && c==-2)  result -= 1.0f/12.0f/FD_cell_size_Z/FD_cell_size_Z;
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
        
/*        if (r2_int<400){
          for (int cnta=0; cnta<10; cnta++)
          for (int cntb=0; cntb<10; cntb++)
          for (int cntc=0; cntc<10; cntc++){
          float y1 = (j + 0.5f) * FD_cell_size_Y + dev_qd_P_10_Y[cntb];
          float y2 = (j - 0.5f) * FD_cell_size_Y + dev_qd_P_10_Y[cntb];
            for (int cnt1=0; cnt1<10; cnt1++){
              float x = i * FD_cell_size_X + dev_qd_P_10_X[cnt1] + dev_qd_P_10_X[cnta];
              for (int cnt3=0; cnt3<10; cnt3++){
                float z = k * FD_cell_size_Z + dev_qd_P_10_Z[cnt3] + dev_qd_P_10_Z[cntc];
                result += 1.0f/8.0f * FD_cell_size_X * FD_cell_size_Z / 4.0f * dev_qd_W_10[cnt1] * dev_qd_W_10[cnt3] *
                  ( z*__powf(x*x+y1*y1+z*z, -1.5f) - z*__powf(x*x+y2*y2+z*z, -1.5f));
              }
            }
          }
        }
        else{
          float r2 = (i*FD_cell_size_X)*(i*FD_cell_size_X) + (j*FD_cell_size_Y)*(j*FD_cell_size_Y) + (k*FD_cell_size_Z)*(k*FD_cell_size_Z);
          result += FD_cell_size_X * FD_cell_size_Y * FD_cell_size_Z * 
                    ( - 3.0f* (j*FD_cell_size_Y) * (k*FD_cell_size_Z) * __powf(r2,-2.5f));
        }*/
        
        
      }
      result *= -1.0f/4.0f/3.14159265f*dim_inverse;
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
       
/*        if (r2_int<400){
          for (int cnta=0; cnta<10; cnta++)
          for (int cntb=0; cntb<10; cntb++)
          for (int cntc=0; cntc<10; cntc++){
          float z1 = (k + 0.5f) * FD_cell_size_Z + dev_qd_P_10_Z[cntc];
          float z2 = (k - 0.5f) * FD_cell_size_Z + dev_qd_P_10_Z[cntc];
            for (int cnt1=0; cnt1<10; cnt1++){
              float x = i * FD_cell_size_X + dev_qd_P_10_X[cnt1] + dev_qd_P_10_X[cnta];
              for (int cnt2=0; cnt2<10; cnt2++){
                float y = j * FD_cell_size_Y + dev_qd_P_10_Y[cnt2] + dev_qd_P_10_Y[cntb];
                result += 1.0f/8.0f * FD_cell_size_X * FD_cell_size_Y / 4.0f * dev_qd_W_10[cnt1] * dev_qd_W_10[cnt2] *
                  ( z1*__powf(x*x+y*y+z1*z1, -1.5f) - z2*__powf(x*x+y*y+z2*z2, -1.5f));
              }
            }
          }
        }
        else{
          float r2 = (i*FD_cell_size_X)*(i*FD_cell_size_X) + (j*FD_cell_size_Y)*(j*FD_cell_size_Y) + (k*FD_cell_size_Z)*(k*FD_cell_size_Z);
          result += FD_cell_size_X * FD_cell_size_Y * FD_cell_size_Z * 
                    (1.0f/ powf(r2,1.5f) - 3.0f* (k*FD_cell_size_Z) * (k*FD_cell_size_Z) * __powf(r2,-2.5f));
        }*/
       
      }
      result *= -1.0f/4.0f/3.14159265f*dim_inverse;

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
      if (exchType==EXCH_12NGBR && exchInConv_Z==1){
        if (a== 0 && b== 0 && c== 0 && Nkernel_X==1)  result -= 5.0f/2.0f/FD_cell_size_Y/FD_cell_size_Y + 5.0f/2.0f/FD_cell_size_Z/FD_cell_size_Z;
        if (a== 0 && b== 0 && c== 0 && Nkernel_X>1 )  result -= 5.0f/2.0f/FD_cell_size_X/FD_cell_size_X + 5.0f/2.0f/FD_cell_size_Y/FD_cell_size_Y + 5.0f/2.0f/FD_cell_size_Z/FD_cell_size_Z;
        if (a== 1 && b== 0 && c== 0)  result += 4.0f/3.0f/D_cell_size[X]/FD_cell_size_X;
        if (a==-1 && b== 0 && c== 0)  result += 4.0f/3.0f/FD_cell_size_X/FD_cell_size_X;
        if (a== 0 && b== 1 && c== 0)  result += 4.0f/3.0f/FD_cell_size_Y/FD_cell_size_Y;
        if (a== 0 && b==-1 && c== 0)  result += 4.0f/3.0f/FD_cell_size_Y/FD_cell_size_Y;
        if (a== 0 && b== 0 && c== 1)  result += 4.0f/3.0f/FD_cell_size_Z/FD_cell_size_Z;
        if (a== 0 && b== 0 && c==-1)  result += 4.0f/3.0f/FD_cell_size_Z/FD_cell_size_Z;
        if (a== 2 && b== 0 && c== 0)  result -= 1.0f/12.0f/D_cell_size[X]/FD_cell_size_X;
        if (a==-2 && b== 0 && c== 0)  result -= 1.0f/12.0f/FD_cell_size_X/FD_cell_size_X;
        if (a== 0 && b== 2 && c== 0)  result -= 1.0f/12.0f/FD_cell_size_Y/FD_cell_size_Y;
        if (a== 0 && b==-2 && c== 0)  result -= 1.0f/12.0f/FD_cell_size_Y/FD_cell_size_Y;
        if (a== 0 && b== 0 && c== 2)  result -= 1.0f/12.0f/FD_cell_size_Z/FD_cell_size_Z;
        if (a== 0 && b== 0 && c==-2)  result -= 1.0f/12.0f/FD_cell_size_Z/FD_cell_size_Z;
      }
    }
  // ______________________________________________________________________________________________
  
  return( result );       //correct for scaling factor in FFTs
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
    get_Quad_Points_micromag3d(&host_qd_P_10[X*10], std_qd_P_10, 10, -0.5f*FD_cell_size_X, 0.5f*FD_cell_size_X);
    get_Quad_Points_micromag3d(&host_qd_P_10[Y*10], std_qd_P_10, 10, -0.5f*FD_cell_size_Y, 0.5f*FD_cell_size_Y);
    get_Quad_Points_micromag3d(&host_qd_P_10[Z*10], std_qd_P_10, 10, -0.5f*FD_cell_size_Z, 0.5f*FD_cell_size_Z);
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

