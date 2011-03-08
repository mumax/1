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
#include "assert.h"
#include <stdio.h>
#include "cpu_kernel_micromag3d.h"
#include "cpu_fft.h"

#include "../macros.h"

#ifdef __cplusplus
extern "C" {
#endif


void cpu_kernel_micromag3d(int *kernelSize, float *cellsize, int exchType, int *exchInConv, int *repetition){
  
  // check input + allocate tensor on device ______________________________________________________
    int kernelStorageN = kernelSize[X] * kernelSize[Y] * (kernelSize[Z]+2);

    tensor *dev_kernel;
    if (p->size[X]==1)
      dev_kernel = as_tensor(new_cpu_array(4*kernelStorageN/2), 2, 4, kernelStorageN/2);  // only real parts!!
    else
      dev_kernel = as_tensor(new_cpu_array(6*kernelStorageN/2), 2, 6, kernelStorageN/2);  // only real parts!!
  // ______________________________________________________________________________________________


  // initialization Gauss quadrature points for integrations + copy to gpu ________________________
    float *dev_qd_W_10 = (float*)calloc(10, sizeof(float));
    float *dev_qd_P_10 = (float*)calloc(3*10, sizeof(float));
    initialize_Gauss_quadrature_micromag3d(dev_qd_W_10, dev_qd_P_10, cellSize);
  // ______________________________________________________________________________________________
  
  
  // Plan initialization of FFTs and initialization of the kernel _________________________________
    cpuFFT3dPlan* kernel_plan = new_cpuFFT3dPlan(kernelSize, kernelSize);
    cpu_init_and_FFT_Greens_kernel_elements_micromag3d(dev_kernel->list, kernelSize, exchType, exchInConv, cellSize, repetition, dev_qd_P_10, dev_qd_W_10, kernel_plan);
  // ______________________________________________________________________________________________ 
  
  delete_cpuFFT3dPlan(kernel_plan);
  free (dev_qd_W_10);
  free (dev_qd_P_10);

  write_tensor(dev_kernel, stdout);
  
  return;

}



/// remark: number of FD cells in a dimension can not be odd if no zero padding!!
void cpu_init_and_FFT_Greens_kernel_elements_micromag3d(float *dev_kernel, int *kernelSize, int exchType, int *exchInConv, float *FD_cell_size, int *repetition, float *dev_qd_P_10, float *dev_qd_W_10, cpuFFT3dPlan* kernel_plan){

  
  int kernelN = kernelSize[X]*kernelSize[Y]*kernelSize[Z];              // size of a kernel component without zeros
  float *dev_temp1 = new_cpu_array(kernelN);                            // temp array on device for storage of kernel component without zeros (input of fft routine)
  int kernelStorageN = kernelSize[X]*kernelSize[Y]*(kernelSize[Z]+2);   // size of a zero padded kernel component
  float *dev_temp2 = new_cpu_array(kernelStorageN);                     // temp array on device for storage of zero padded kernel component (output of fft routine)
  
 
  // Main function operations _____________________________________________________________________
    int rank0 = 0;                                      // defines the first rank of the Greens kernel, N0>1: [xx, xy, xz, yy, yz, zz], N1=1: [xx, yy, yz, zz]
    for (int co1=0; co1<3; co1++){                      // for a Greens kernel component [co1,co2]:
      for (int co2=co1; co2<3; co2++){
        if (co1==0 && co2>0 && kernelSize[0]==1)  continue;    // N0=1 -> xy and xz components have only zeros, so left out.
          // Put all elements in 'dev_temp1' to zero.
        cpu_zero(dev_temp1, kernelN);    
          // Fill in the elements.
        _cpu_init_Greens_kernel_elements_micromag3d(dev_temp1, kernelSize, exchType, exchInConv, co1, co2, FD_cell_size, repetition, dev_qd_P_10, dev_qd_W_10);
          // Fourier transform the kernel component.
        cpuFFT3dPlan_forward(kernel_plan, dev_temp1, dev_temp2);
          // Copy the real parts to the corresponding place in the dev_kernel tensor.
        _cpu_extract_real_parts_micromag3d(&dev_kernel[rank0*kernelStorageN/2], dev_temp2, N2);
        rank0++;                                        // get ready for next component
      }
    } 
  // ______________________________________________________________________________________________


  free (dev_temp1);
  free (dev_temp2);
  
  return;
}



typedef struct{
  float *FD_cell_size, *dev_temp, *dev_qd_P_10, *dev_qd_W_10;
  int *Nkernel, *exchInConv, *repetition; 
  int exchType, co1, co2;
} _cpu_init_Greens_kernel_elements_micromag3d_arg;

float _cpu_get_Greens_element_micromag3d(_cpu_init_Greens_kernel_elements_micromag3d_arg *arg, int a, int b, int c){

  int *Nkernel = arg->Nkernel;
  int exchType = arg->exchType;
  int *exchInConv = arg->exchInConv;
  int co1 = arg->co1;
  int co2 = arg->co2;
  float *FD_cell_size = arg->FD_cell_size;
  int *repetition = arg->repetition;
  float *dev_qd_P_10 = arg->dev_qd_P_10;
  float *dev_qd_W_10 = arg->dev_qd_W_10;
  
  float result = 0.0f;
  float *dev_qd_P_10_X = &dev_qd_P_10[X*10];
  float *dev_qd_P_10_Y = &dev_qd_P_10[Y*10];
  float *dev_qd_P_10_Z = &dev_qd_P_10[Z*10];
  float dim_inverse = 1.0f/( (float) Nkernel[X]*Nkernel[Y]*Nkernel[Z] );
  
  // for elements in Kernel component gxx _________________________________________________________
    if (co1==0 && co2==0){

      for(int cnta=-repetition[X]; cnta<=repetition[X]; cnta++)
      for(int cntb=-repetition[Y]; cntb<=repetition[Y]; cntb++)
      for(int cntc=-repetition[Z]; cntc<=repetition[Z]; cntc++){

        int i = a + cnta*Nkernel[X]/2;
        int j = b + cntb*Nkernel[Y]/2;
        int k = c + cntc*Nkernel[Z]/2;
        int r2_int = i*i+j*j+k*k;

        if (r2_int<400){
          float x1 = (i + 0.5f) * FD_cell_size[X];
          float x2 = (i - 0.5f) * FD_cell_size[X];
          for (int cnt2=0; cnt2<10; cnt2++){
            float y = j * FD_cell_size[Y] + dev_qd_P_10_Y[cnt2];
            for (int cnt3=0; cnt3<10; cnt3++){
              float z = k * FD_cell_size[Z] + dev_qd_P_10_Z[cnt3];
              result += FD_cell_size[Y] * FD_cell_size[Z] / 4.0f * dev_qd_W_10[cnt2] * dev_qd_W_10[cnt3] *
                ( x1*powf(x1*x1+y*y+z*z, -1.5f) - x2*powf(x2*x2+y*y+z*z, -1.5f));
            }
          }
        }
        else{
          float r2 = (i*FD_cell_size[X])*(i*FD_cell_size[X]) + (j*FD_cell_size[Y])*(j*FD_cell_size[Y]) + (k*FD_cell_size[Z])*(k*FD_cell_size[Z]);
          result += FD_cell_size[X] * FD_cell_size[Y] * FD_cell_size[Z] *
                    (1.0f/ powf(r2,1.5f) - 3.0f* (i*FD_cell_size[X]) * (i*FD_cell_size[X]) * powf(r2,-2.5f));
        }
        
      
/*        if (r2_int<400){   // The source is considered as a uniformly magnetized FD cell and the field is averaged over the complete observation FD cell
          for (int cnta=0; cnta<10; cnta++)
          for (int cntb=0; cntb<10; cntb++)
          for (int cntc=0; cntc<10; cntc++){
            float x1 = (i + 0.5f) * FD_cell_size[X] + dev_qd_P_10_X[cnta];
            float x2 = (i - 0.5f) * FD_cell_size[X] + dev_qd_P_10_X[cnta];
            for (int cnt2=0; cnt2<10; cnt2++){
              float y = j * FD_cell_size[Y] + dev_qd_P_10_Y[cnt2] + dev_qd_P_10_Y[cntb];
              for (int cnt3=0; cnt3<10; cnt3++){
                float z = k * FD_cell_size[Z] + dev_qd_P_10_Z[cnt3] + dev_qd_P_10_Z[cntc];
                result += 1.0f/8.0f * FD_cell_size[Y] * FD_cell_size[Z] / 4.0f * dev_qd_W_10[cnt2] * dev_qd_W_10[cnt3] *
                  ( x1*powf(x1*x1+y*y+z*z, -1.5f) - x2*powf(x2*x2+y*y+z*z, -1.5f));
              }
            }
          }
        }
        else{   // The source is considered as a point. No averaging of the field in the observation FD cell
          float r2 = (i*FD_cell_size[X])*(i*FD_cell_size[X]) + (j*FD_cell_size[Y])*(j*FD_cell_size[Y]) + (k*FD_cell_size[Z])*(k*FD_cell_size[Z]);
          result += FD_cell_size[X] * FD_cell_size[Y] * FD_cell_size[Z] *
                    (1.0f/ powf(r2,1.5f) - 3.0f* (i*FD_cell_size[X]) * (i*FD_cell_size[X]) * powf(r2,-2.5f));
        }*/
        
      
      }
      result *= -1.0f/4.0f/3.14159265f*dim_inverse;

      if (exchType==EXCH_6NGBR && exchInConv[X]==1){
        if (a== 0 && b== 0 && c== 0 && Nkernel[X]==1)  result -= 2.0f/FD_cell_size[Y]/FD_cell_size[Y] + 2.0f/FD_cell_size[Z]/FD_cell_size[Z];
        if (a== 0 && b== 0 && c== 0 && Nkernel[X]>1 )  result -= 2.0f/FD_cell_size[X]/FD_cell_size[X] + 2.0f/FD_cell_size[Y]/FD_cell_size[Y] + 2.0f/FD_cell_size[Z]/FD_cell_size[Z];
        if (a== 1 && b== 0 && c== 0)  result += 1.0f/FD_cell_size[X]/FD_cell_size[X];
        if (a==-1 && b== 0 && c== 0)  result += 1.0f/FD_cell_size[X]/FD_cell_size[X];
        if (a== 0 && b== 1 && c== 0)  result += 1.0f/FD_cell_size[Y]/FD_cell_size[Y];
        if (a== 0 && b==-1 && c== 0)  result += 1.0f/FD_cell_size[Y]/FD_cell_size[Y];
        if (a== 0 && b== 0 && c== 1)  result += 1.0f/FD_cell_size[Z]/FD_cell_size[Z];
        if (a== 0 && b== 0 && c==-1)  result += 1.0f/FD_cell_size[Z]/FD_cell_size[Z];
      }
      if (exchType==EXCH_12NGBR && exchInConv[X]==1){
        if (a== 0 && b== 0 && c== 0 && Nkernel[X]==1)  result -= 5.0f/2.0f/FD_cell_size[Y]/FD_cell_size[Y] + 5.0f/2.0f/FD_cell_size[Z]/FD_cell_size[Z];
        if (a== 0 && b== 0 && c== 0 && Nkernel[X]>1 )  result -= 5.0f/2.0f/FD_cell_size[X]/FD_cell_size[X] + 5.0f/2.0f/FD_cell_size[Y]/FD_cell_size[Y] + 5.0f/2.0f/FD_cell_size[Z]/FD_cell_size[Z];
        if (a== 1 && b== 0 && c== 0)  result += 4.0f/3.0f/D_cell_size[X]/FD_cell_size[X];
        if (a==-1 && b== 0 && c== 0)  result += 4.0f/3.0f/FD_cell_size[X]/FD_cell_size[X];
        if (a== 0 && b== 1 && c== 0)  result += 4.0f/3.0f/FD_cell_size[Y]/FD_cell_size[Y];
        if (a== 0 && b==-1 && c== 0)  result += 4.0f/3.0f/FD_cell_size[Y]/FD_cell_size[Y];
        if (a== 0 && b== 0 && c== 1)  result += 4.0f/3.0f/FD_cell_size[Z]/FD_cell_size[Z];
        if (a== 0 && b== 0 && c==-1)  result += 4.0f/3.0f/FD_cell_size[Z]/FD_cell_size[Z];
        if (a== 2 && b== 0 && c== 0)  result -= 1.0f/12.0f/D_cell_size[X]/FD_cell_size[X];
        if (a==-2 && b== 0 && c== 0)  result -= 1.0f/12.0f/FD_cell_size[X]/FD_cell_size[X];
        if (a== 0 && b== 2 && c== 0)  result -= 1.0f/12.0f/FD_cell_size[Y]/FD_cell_size[Y];
        if (a== 0 && b==-2 && c== 0)  result -= 1.0f/12.0f/FD_cell_size[Y]/FD_cell_size[Y];
        if (a== 0 && b== 0 && c== 2)  result -= 1.0f/12.0f/FD_cell_size[Z]/FD_cell_size[Z];
        if (a== 0 && b== 0 && c==-2)  result -= 1.0f/12.0f/FD_cell_size[Z]/FD_cell_size[Z];
      }
      
    }
  // ______________________________________________________________________________________________


  // for elements in Kernel component gxy _________________________________________________________
    if (co1==0 && co2==1){
      for(int cnta=-repetition[X]; cnta<=repetition[X]; cnta++)
      for(int cntb=-repetition[Y]; cntb<=repetition[Y]; cntb++)
      for(int cntc=-repetition[Z]; cntc<=repetition[Z]; cntc++){

        int i = a + cnta*Nkernel[X]/2;
        int j = b + cntb*Nkernel[Y]/2;
        int k = c + cntc*Nkernel[Z]/2;
        int r2_int = i*i+j*j+k*k;

        if (r2_int<400){
          float x1 = (i + 0.5f) * FD_cell_size[X];
          float x2 = (i - 0.5f) * FD_cell_size[X];
          for (int cnt2=0; cnt2<10; cnt2++){
            float y = j * FD_cell_size[Y] + dev_qd_P_10_Y[cnt2];
            for (int cnt3=0; cnt3<10; cnt3++){
              float z = k * FD_cell_size[Z] + dev_qd_P_10_Z[cnt3];
              result += 1.0f/8.0f * FD_cell_size[Y] * FD_cell_size[Z] / 4.0f * dev_qd_W_10[cnt2] * dev_qd_W_10[cnt3] *
                ( y*powf(x1*x1+y*y+z*z, -1.5f) - y*powf(x2*x2+y*y+z*z, -1.5f));
            }
          }
        }
        else{
          float r2 = (i*FD_cell_size[X])*(i*FD_cell_size[X]) + (j*FD_cell_size[Y])*(j*FD_cell_size[Y]) + (k*FD_cell_size[Z])*(k*FD_cell_size[Z]);
          result += FD_cell_size[X] * FD_cell_size[Y] * FD_cell_size[Z] * 
                    (- 3.0f* (i*FD_cell_size[X]) * (j*FD_cell_size[Y]) * powf(r2,-2.5f));
        }
       
//         if (r2_int<400){
//           for (int cnta=0; cnta<10; cnta++)
//           for (int cntb=0; cntb<10; cntb++)
//           for (int cntc=0; cntc<10; cntc++){
//             float x1 = (i + 0.5f) * FD_cell_size[X] + dev_qd_P_10_X[cnta];
//             float x2 = (i - 0.5f) * FD_cell_size[X] + dev_qd_P_10_X[cnta];
//             for (int cnt2=0; cnt2<10; cnt2++){
//               float y = j * FD_cell_size[Y] + dev_qd_P_10_Y[cnt2] + dev_qd_P_10_Y[cntb];
//               for (int cnt3=0; cnt3<10; cnt3++){
//                 float z = k * FD_cell_size[Z] + dev_qd_P_10_Z[cnt3] + dev_qd_P_10_Z[cntc];
//                 result += 1.0f/8.0f * FD_cell_size[Y] * FD_cell_size[Z] / 4.0f * dev_qd_W_10[cnt2] * dev_qd_W_10[cnt3] *
//                   ( y*powf(x1*x1+y*y+z*z, -1.5f) - y*powf(x2*x2+y*y+z*z, -1.5f));
//               }
//             }
//           }
//         }
//         else{
//           float r2 = (i*FD_cell_size[X])*(i*FD_cell_size[X]) + (j*FD_cell_size[Y])*(j*FD_cell_size[Y]) + (k*FD_cell_size[Z])*(k*FD_cell_size[Z]);
//           result += FD_cell_size[X] * FD_cell_size[Y] * FD_cell_size[Z] * 
//                     (- 3.0f* (i*FD_cell_size[X]) * (j*FD_cell_size[Y]) * powf(r2,-2.5f));
//         }

      }
      result *= -1.0f/4.0f/3.14159265f*dim_inverse;
    }
  // ______________________________________________________________________________________________


  // for elements in Kernel component gxz _________________________________________________________
    if (co1==0 && co2==2){
      for(int cnta=-repetition[X]; cnta<=repetition[X]; cnta++)
      for(int cntb=-repetition[Y]; cntb<=repetition[Y]; cntb++)
      for(int cntc=-repetition[Z]; cntc<=repetition[Z]; cntc++){

        int i = a + cnta*Nkernel[X]/2;
        int j = b + cntb*Nkernel[Y]/2;
        int k = c + cntc*Nkernel[Z]/2;
        int r2_int = i*i+j*j+k*k;

        if (r2_int<400){
          float x1 = (i + 0.5f) * FD_cell_size[X];
          float x2 = (i - 0.5f) * FD_cell_size[X];
          for (int cnt2=0; cnt2<10; cnt2++){
            float y = j * FD_cell_size[Y] + dev_qd_P_10_Y[cnt2];
            for (int cnt3=0; cnt3<10; cnt3++){
              float z = k * FD_cell_size[Z] + dev_qd_P_10_Z[cnt3];
              result += FD_cell_size[Y] * FD_cell_size[Z] / 4.0f * dev_qd_W_10[cnt2] * dev_qd_W_10[cnt3] *
                ( z*powf(x1*x1+y*y+z*z, -1.5f) - z*powf(x2*x2+y*y+z*z, -1.5f));
            }
          }
        }
        else{
          float r2 = (i*FD_cell_size[X])*(i*FD_cell_size[X]) + (j*FD_cell_size[Y])*(j*FD_cell_size[Y]) + (k*FD_cell_size[Z])*(k*FD_cell_size[Z]);
          result += FD_cell_size[X] * FD_cell_size[Y] * FD_cell_size[Z] * 
                    (- 3.0f* (i*FD_cell_size[X]) * (k*FD_cell_size[Y]) * powf(r2,-2.5f));
        }

/*        if (r2_int<400){
          for (int cnta=0; cnta<10; cnta++)
          for (int cntb=0; cntb<10; cntb++)
          for (int cntc=0; cntc<10; cntc++){
            float x1 = (i + 0.5f) * FD_cell_size[X] + dev_qd_P_10_X[cnta];
            float x2 = (i - 0.5f) * FD_cell_size[X] + dev_qd_P_10_X[cnta];
            for (int cnt2=0; cnt2<10; cnt2++){
              float y = j * FD_cell_size[Y] + dev_qd_P_10_Y[cnt2] + dev_qd_P_10_Y[cntb];
              for (int cnt3=0; cnt3<10; cnt3++){
                float z = k * FD_cell_size[Z] + dev_qd_P_10_Z[cnt3] + dev_qd_P_10_Z[cntc];
                result += 1.0f/8.0f * FD_cell_size[Y] * FD_cell_size[Z] / 4.0f * dev_qd_W_10[cnt2] * dev_qd_W_10[cnt3] *
                  ( z*powf(x1*x1+y*y+z*z, -1.5f) - z*powf(x2*x2+y*y+z*z, -1.5f));
              }
            }
          }
        }
        else{
          float r2 = (i*FD_cell_size[X])*(i*FD_cell_size[X]) + (j*FD_cell_size[Y])*(j*FD_cell_size[Y]) + (k*FD_cell_size[Z])*(k*FD_cell_size[Z]);
          result += FD_cell_size[X] * FD_cell_size[Y] * FD_cell_size[Z] * 
                    (- 3.0f* (i*FD_cell_size[X]) * (k*FD_cell_size[Y]) * powf(r2,-2.5f));
        }*/
      
      
      }
      result *= -1.0f/4.0f/3.14159265f*dim_inverse;
    }
  // ______________________________________________________________________________________________


  // for elements in Kernel component gyy _________________________________________________________
    if (co1==1 && co2==1){
      for(int cnta=-repetition[X]; cnta<=repetition[X]; cnta++)
      for(int cntb=-repetition[Y]; cntb<=repetition[Y]; cntb++)
      for(int cntc=-repetition[Z]; cntc<=repetition[Z]; cntc++){

        int i = a + cnta*Nkernel[X]/2;
        int j = b + cntb*Nkernel[Y]/2;
        int k = c + cntc*Nkernel[Z]/2;
        int r2_int = i*i+j*j+k*k;

        if (r2_int<400){
          float y1 = (j + 0.5f) * FD_cell_size[Y];
          float y2 = (j - 0.5f) * FD_cell_size[Y];
          for (int cnt1=0; cnt1<10; cnt1++){
            float x = i * FD_cell_size[X] + dev_qd_P_10_X[cnt1];
            for (int cnt3=0; cnt3<10; cnt3++){
              float z = k * FD_cell_size[Z] + dev_qd_P_10_Z[cnt3];
              result += FD_cell_size[X] * FD_cell_size[Z] / 4.0f * dev_qd_W_10[cnt1] * dev_qd_W_10[cnt3] *
                ( y1*powf(x*x+y1*y1+z*z, -1.5f) - y2*powf(x*x+y2*y2+z*z, -1.5f));
            }
          }
        }
        else{
          float r2 = (i*FD_cell_size[X])*(i*FD_cell_size[X]) + (j*FD_cell_size[Y])*(j*FD_cell_size[Y]) + (k*FD_cell_size[Z])*(k*FD_cell_size[Z]);
          result += FD_cell_size[X] * FD_cell_size[Y] * FD_cell_size[Z] * 
                    (1.0f/ powf(r2,1.5f) - 3.0f* (j*FD_cell_size[Y]) * (j*FD_cell_size[Y]) * powf(r2,-2.5f));
        }

      
/*        if (r2_int<400){
          for (int cnta=0; cnta<10; cnta++)
          for (int cntb=0; cntb<10; cntb++)
          for (int cntc=0; cntc<10; cntc++){
            float y1 = (j + 0.5f) * FD_cell_size[Y] + dev_qd_P_10_Y[cntb];
            float y2 = (j - 0.5f) * FD_cell_size[Y] + dev_qd_P_10_Y[cntb];
            for (int cnt1=0; cnt1<10; cnt1++){
              float x = i * FD_cell_size[X] + dev_qd_P_10_X[cnt1] + dev_qd_P_10_X[cnta];
              for (int cnt3=0; cnt3<10; cnt3++){
                float z = k * FD_cell_size[Z] + dev_qd_P_10_Z[cnt3] + dev_qd_P_10_Z[cntc];
                result += 1.0f/8.0f * FD_cell_size[X] * FD_cell_size[Z] / 4.0f * dev_qd_W_10[cnt1] * dev_qd_W_10[cnt3] *
                  ( y1*powf(x*x+y1*y1+z*z, -1.5f) - y2*powf(x*x+y2*y2+z*z, -1.5f));
              }
            }
          }
        }
        else{
          float r2 = (i*FD_cell_size[X])*(i*FD_cell_size[X]) + (j*FD_cell_size[Y])*(j*FD_cell_size[Y]) + (k*FD_cell_size[Z])*(k*FD_cell_size[Z]);
          result += FD_cell_size[X] * FD_cell_size[Y] * FD_cell_size[Z] * 
                    (1.0f/ powf(r2,1.5f) - 3.0f* (j*FD_cell_size[Y]) * (j*FD_cell_size[Y]) * powf(r2,-2.5f));
        }*/
      
      
      }
      result *= -1.0f/4.0f/3.14159265f*dim_inverse;

      if (exchType==EXCH_6NGBR && exchInConv[Y]==1){
        if (a== 0 && b== 0 && c== 0 && Nkernel[X]==1)  result -= 2.0f/FD_cell_size[Y]/FD_cell_size[Y] + 2.0f/FD_cell_size[Z]/FD_cell_size[Z];
        if (a== 0 && b== 0 && c== 0 && Nkernel[X]>1 )  result -= 2.0f/FD_cell_size[X]/FD_cell_size[X] + 2.0f/FD_cell_size[Y]/FD_cell_size[Y] + 2.0f/FD_cell_size[Z]/FD_cell_size[Z];
        if (a== 1 && b== 0 && c== 0)  result += 1.0f/FD_cell_size[X]/FD_cell_size[X];
        if (a==-1 && b== 0 && c== 0)  result += 1.0f/FD_cell_size[X]/FD_cell_size[X];
        if (a== 0 && b== 1 && c== 0)  result += 1.0f/FD_cell_size[Y]/FD_cell_size[Y];
        if (a== 0 && b==-1 && c== 0)  result += 1.0f/FD_cell_size[Y]/FD_cell_size[Y];
        if (a== 0 && b== 0 && c== 1)  result += 1.0f/FD_cell_size[Z]/FD_cell_size[Z];
        if (a== 0 && b== 0 && c==-1)  result += 1.0f/FD_cell_size[Z]/FD_cell_size[Z];
      }
      if (exchType==EXCH_12NGBR && exchInConv[Y]==1){
        if (a== 0 && b== 0 && c== 0 && Nkernel[X]==1)  result -= 5.0f/2.0f/FD_cell_size[Y]/FD_cell_size[Y] + 5.0f/2.0f/FD_cell_size[Z]/FD_cell_size[Z];
        if (a== 0 && b== 0 && c== 0 && Nkernel[X]>1 )  result -= 5.0f/2.0f/FD_cell_size[X]/FD_cell_size[X] + 5.0f/2.0f/FD_cell_size[Y]/FD_cell_size[Y] + 5.0f/2.0f/FD_cell_size[Z]/FD_cell_size[Z];
        if (a== 1 && b== 0 && c== 0)  result += 4.0f/3.0f/D_cell_size[X]/FD_cell_size[X];
        if (a==-1 && b== 0 && c== 0)  result += 4.0f/3.0f/FD_cell_size[X]/FD_cell_size[X];
        if (a== 0 && b== 1 && c== 0)  result += 4.0f/3.0f/FD_cell_size[Y]/FD_cell_size[Y];
        if (a== 0 && b==-1 && c== 0)  result += 4.0f/3.0f/FD_cell_size[Y]/FD_cell_size[Y];
        if (a== 0 && b== 0 && c== 1)  result += 4.0f/3.0f/FD_cell_size[Z]/FD_cell_size[Z];
        if (a== 0 && b== 0 && c==-1)  result += 4.0f/3.0f/FD_cell_size[Z]/FD_cell_size[Z];
        if (a== 2 && b== 0 && c== 0)  result -= 1.0f/12.0f/D_cell_size[X]/FD_cell_size[X];
        if (a==-2 && b== 0 && c== 0)  result -= 1.0f/12.0f/FD_cell_size[X]/FD_cell_size[X];
        if (a== 0 && b== 2 && c== 0)  result -= 1.0f/12.0f/FD_cell_size[Y]/FD_cell_size[Y];
        if (a== 0 && b==-2 && c== 0)  result -= 1.0f/12.0f/FD_cell_size[Y]/FD_cell_size[Y];
        if (a== 0 && b== 0 && c== 2)  result -= 1.0f/12.0f/FD_cell_size[Z]/FD_cell_size[Z];
        if (a== 0 && b== 0 && c==-2)  result -= 1.0f/12.0f/FD_cell_size[Z]/FD_cell_size[Z];
      }
      
    }
// ______________________________________________________________________________________________


  // for elements in Kernel component gyz _________________________________________________________
    if (co1==1 && co2==2){
      for(int cnta=-repetition[X]; cnta<=repetition[X]; cnta++)
      for(int cntb=-repetition[Y]; cntb<=repetition[Y]; cntb++)
      for(int cntc=-repetition[Z]; cntc<=repetition[Z]; cntc++){

        int i = a + cnta*Nkernel[X]/2;
        int j = b + cntb*Nkernel[Y]/2;
        int k = c + cntc*Nkernel[Z]/2;
        int r2_int = i*i+j*j+k*k;

        if (r2_int<400){
          float y1 = (j + 0.5f) * FD_cell_size[Y];
          float y2 = (j - 0.5f) * FD_cell_size[Y];
          for (int cnt1=0; cnt1<10; cnt1++){
            float x = i * FD_cell_size[X] + dev_qd_P_10_X[cnt1];
            for (int cnt3=0; cnt3<10; cnt3++){
              float z = k * FD_cell_size[Z] + dev_qd_P_10_Z[cnt3];
              result += FD_cell_size[X] * FD_cell_size[Z] / 4.0f * dev_qd_W_10[cnt1] * dev_qd_W_10[cnt3] *
                ( z*powf(x*x+y1*y1+z*z, -1.5f) - z*powf(x*x+y2*y2+z*z, -1.5f));
            }
          }
        }
        else{
          float r2 = (i*FD_cell_size[X])*(i*FD_cell_size[X]) + (j*FD_cell_size[Y])*(j*FD_cell_size[Y]) + (k*FD_cell_size[Z])*(k*FD_cell_size[Z]);
          result += FD_cell_size[X] * FD_cell_size[Y] * FD_cell_size[Z] * 
                    ( - 3.0f* (j*FD_cell_size[Y]) * (k*FD_cell_size[Z]) * powf(r2,-2.5f));
        }

/*        if (r2_int<400){
          for (int cnta=0; cnta<10; cnta++)
          for (int cntb=0; cntb<10; cntb++)
          for (int cntc=0; cntc<10; cntc++){
          float y1 = (j + 0.5f) * FD_cell_size[Y] + dev_qd_P_10_Y[cntb];
          float y2 = (j - 0.5f) * FD_cell_size[Y] + dev_qd_P_10_Y[cntb];
            for (int cnt1=0; cnt1<10; cnt1++){
              float x = i * FD_cell_size[X] + dev_qd_P_10_X[cnt1] + dev_qd_P_10_X[cnta];
              for (int cnt3=0; cnt3<10; cnt3++){
                float z = k * FD_cell_size[Z] + dev_qd_P_10_Z[cnt3] + dev_qd_P_10_Z[cntc];
                result += 1.0f/8.0f * FD_cell_size[X] * FD_cell_size[Z] / 4.0f * dev_qd_W_10[cnt1] * dev_qd_W_10[cnt3] *
                  ( z*powf(x*x+y1*y1+z*z, -1.5f) - z*powf(x*x+y2*y2+z*z, -1.5f));
              }
            }
          }
        }
        else{
          float r2 = (i*FD_cell_size[X])*(i*FD_cell_size[X]) + (j*FD_cell_size[Y])*(j*FD_cell_size[Y]) + (k*FD_cell_size[Z])*(k*FD_cell_size[Z]);
          result += FD_cell_size[X] * FD_cell_size[Y] * FD_cell_size[Z] * 
                    ( - 3.0f* (j*FD_cell_size[Y]) * (k*FD_cell_size[Z]) * powf(r2,-2.5f));
        }*/
      
      
      }
      result *= -1.0f/4.0f/3.14159265f*dim_inverse;
    }
  // ______________________________________________________________________________________________


  // for elements in Kernel component gzz _________________________________________________________
    if (co1==2 && co2==2){
      for(int cnta=-repetition[X]; cnta<=repetition[X]; cnta++)
      for(int cntb=-repetition[Y]; cntb<=repetition[Y]; cntb++)
      for(int cntc=-repetition[Z]; cntc<=repetition[Z]; cntc++){

        int i = a + cnta*Nkernel[X]/2;
        int j = b + cntb*Nkernel[Y]/2;
        int k = c + cntc*Nkernel[Z]/2;
        int r2_int = i*i+j*j+k*k;

        if (r2_int<400){
          float z1 = (k + 0.5f) * FD_cell_size[Z];
          float z2 = (k - 0.5f) * FD_cell_size[Z];
          for (int cnt1=0; cnt1<10; cnt1++){
            float x = i * FD_cell_size[X] + dev_qd_P_10_X[cnt1];
            for (int cnt2=0; cnt2<10; cnt2++){
              float y = j * FD_cell_size[Y] + dev_qd_P_10_Y[cnt2];
              result += FD_cell_size[X] * FD_cell_size[Y] / 4.0f * dev_qd_W_10[cnt1] * dev_qd_W_10[cnt2] *
                ( z1*powf(x*x+y*y+z1*z1, -1.5f) - z2*powf(x*x+y*y+z2*z2, -1.5f));
            }
          }
        }
        else{
          float r2 = (i*FD_cell_size[X])*(i*FD_cell_size[X]) + (j*FD_cell_size[Y])*(j*FD_cell_size[Y]) + (k*FD_cell_size[Z])*(k*FD_cell_size[Z]);
          result += FD_cell_size[X] * FD_cell_size[Y] * FD_cell_size[Z] * 
                    (1.0f/ powf(r2,1.5f) - 3.0f* (k*FD_cell_size[Z]) * (k*FD_cell_size[Z]) * powf(r2,-2.5f));
        }

/*        if (r2_int<400){
          for (int cnta=0; cnta<10; cnta++)
          for (int cntb=0; cntb<10; cntb++)
          for (int cntc=0; cntc<10; cntc++){
          float z1 = (k + 0.5f) * FD_cell_size[Z] + dev_qd_P_10_Z[cntc];
          float z2 = (k - 0.5f) * FD_cell_size[Z] + dev_qd_P_10_Z[cntc];
            for (int cnt1=0; cnt1<10; cnt1++){
              float x = i * FD_cell_size[X] + dev_qd_P_10_X[cnt1] + dev_qd_P_10_X[cnta];
              for (int cnt2=0; cnt2<10; cnt2++){
                float y = j * FD_cell_size[Y] + dev_qd_P_10_Y[cnt2] + dev_qd_P_10_Y[cntb];
                result += 1.0f/8.0f * FD_cell_size[X] * FD_cell_size[Y] / 4.0f * dev_qd_W_10[cnt1] * dev_qd_W_10[cnt2] *
                  ( z1*powf(x*x+y*y+z1*z1, -1.5f) - z2*powf(x*x+y*y+z2*z2, -1.5f));
              }
            }
          }
        }
        else{
          float r2 = (i*FD_cell_size[X])*(i*FD_cell_size[X]) + (j*FD_cell_size[Y])*(j*FD_cell_size[Y]) + (k*FD_cell_size[Z])*(k*FD_cell_size[Z]);
          result += FD_cell_size[X] * FD_cell_size[Y] * FD_cell_size[Z] * 
                    (1.0f/ powf(r2,1.5f) - 3.0f* (k*FD_cell_size[Z]) * (k*FD_cell_size[Z]) * powf(r2,-2.5f));
        }*/
      
      }
      result *= -1.0f/4.0f/3.14159265f*dim_inverse;

      if (exchType==EXCH_6NGBR && exchInConv[Z]==1){
        if (a== 0 && b== 0 && c== 0 && Nkernel[X]==1)  result -= 2.0f/FD_cell_size[Y]/FD_cell_size[Y] + 2.0f/FD_cell_size[Z]/FD_cell_size[Z];
        if (a== 0 && b== 0 && c== 0 && Nkernel[X]>1 )  result -= 2.0f/FD_cell_size[X]/FD_cell_size[X] + 2.0f/FD_cell_size[Y]/FD_cell_size[Y] + 2.0f/FD_cell_size[Z]/FD_cell_size[Z];
        if (a== 1 && b== 0 && c== 0)  result += 1.0f/FD_cell_size[X]/FD_cell_size[X];
        if (a==-1 && b== 0 && c== 0)  result += 1.0f/FD_cell_size[X]/FD_cell_size[X];
        if (a== 0 && b== 1 && c== 0)  result += 1.0f/FD_cell_size[Y]/FD_cell_size[Y];
        if (a== 0 && b==-1 && c== 0)  result += 1.0f/FD_cell_size[Y]/FD_cell_size[Y];
        if (a== 0 && b== 0 && c== 1)  result += 1.0f/FD_cell_size[Z]/FD_cell_size[Z];
        if (a== 0 && b== 0 && c==-1)  result += 1.0f/FD_cell_size[Z]/FD_cell_size[Z];
      }
      if (exchType==EXCH_12NGBR && exchInConv[Z]==1){
        if (a== 0 && b== 0 && c== 0 && Nkernel[X]==1)  result -= 5.0f/2.0f/FD_cell_size[Y]/FD_cell_size[Y] + 5.0f/2.0f/FD_cell_size[Z]/FD_cell_size[Z];
        if (a== 0 && b== 0 && c== 0 && Nkernel[X]>1 )  result -= 5.0f/2.0f/FD_cell_size[X]/FD_cell_size[X] + 5.0f/2.0f/FD_cell_size[Y]/FD_cell_size[Y] + 5.0f/2.0f/FD_cell_size[Z]/FD_cell_size[Z];
        if (a== 1 && b== 0 && c== 0)  result += 4.0f/3.0f/D_cell_size[X]/FD_cell_size[X];
        if (a==-1 && b== 0 && c== 0)  result += 4.0f/3.0f/FD_cell_size[X]/FD_cell_size[X];
        if (a== 0 && b== 1 && c== 0)  result += 4.0f/3.0f/FD_cell_size[Y]/FD_cell_size[Y];
        if (a== 0 && b==-1 && c== 0)  result += 4.0f/3.0f/FD_cell_size[Y]/FD_cell_size[Y];
        if (a== 0 && b== 0 && c== 1)  result += 4.0f/3.0f/FD_cell_size[Z]/FD_cell_size[Z];
        if (a== 0 && b== 0 && c==-1)  result += 4.0f/3.0f/FD_cell_size[Z]/FD_cell_size[Z];
        if (a== 2 && b== 0 && c== 0)  result -= 1.0f/12.0f/D_cell_size[X]/FD_cell_size[X];
        if (a==-2 && b== 0 && c== 0)  result -= 1.0f/12.0f/FD_cell_size[X]/FD_cell_size[X];
        if (a== 0 && b== 2 && c== 0)  result -= 1.0f/12.0f/FD_cell_size[Y]/FD_cell_size[Y];
        if (a== 0 && b==-2 && c== 0)  result -= 1.0f/12.0f/FD_cell_size[Y]/FD_cell_size[Y];
        if (a== 0 && b== 0 && c== 2)  result -= 1.0f/12.0f/FD_cell_size[Z]/FD_cell_size[Z];
        if (a== 0 && b== 0 && c==-2)  result -= 1.0f/12.0f/FD_cell_size[Z]/FD_cell_size[Z];
      }
    }
  // ______________________________________________________________________________________________
  
  return( result );
}

void _cpu_init_Greens_kernel_elements_micromag3d_t(int id){
  
  _cpu_init_Greens_kernel_elements_micromag3d_arg *arg = (_cpu_init_Greens_kernel_elements_micromag3d_arg *) func_arg;

  int start, stop;
  init_start_stop (&start, &stop, id, (arg->Nkernel[X]+1)/2);

  int N2 = arg->Nkernel[Z];
  int N12 = arg->Nkernel[Y] * N2;
  

  for (int i=start; i<stop; i++)
    for (int j=0; j<arg->Nkernel[Y]/2; j++)
      for (int k=0; k<N2/2; k++){
          arg->dev_temp[                  i*N12 +                   j*N2 +                 k] = _cpu_get_Greens_element_micromag3d(arg, i, j, k);
        if (i>0)
          arg->dev_temp[(arg->Nkernel[X]-i)*N12 +                   j*N2 +                 k] = _cpu_get_Greens_element_micromag3d(arg, -i, j, k);
        if (j>0)
          arg->dev_temp[                  i*N12 + (arg->Nkernel[Y]-j)*N2 +                 k] = _cpu_get_Greens_element_micromag3d(arg, i, -j, k);
        if (k>0) 
          arg->dev_temp[                  i*N12 +                   j*N2 + arg->Nkernel[Z]-k] = _cpu_get_Greens_element_micromag3d(arg, i, j, -k);
        if (i>0 && j>0)
          arg->dev_temp[(arg->Nkernel[X]-i)*N12 + (arg->Nkernel[Y]-j)*N2 +                 k] = _cpu_get_Greens_element_micromag3d(arg, -i, -j, k);
        if (i>0 && k>0) 
          arg->dev_temp[(arg->Nkernel[X]-i)*N12 +                   j*N2 + arg->Nkernel[Z]-k] = _cpu_get_Greens_element_micromag3d(arg, -i, j, -k);
        if (j>0 && k>0) 
          arg->dev_temp[                  i*N12 + (arg->Nkernel[Y]-j)*N2 + arg->Nkernel[Z]-k] = _cpu_get_Greens_element_micromag3d(arg, i, -j, -k);
        if (i>0 && j>0 && k>0) 
          arg->dev_temp[(arg->Nkernel[X]-i)*N12 + (arg->Nkernel[Y]-j)*N2 + arg->Nkernel[Z]-k] = _cpu_get_Greens_element_micromag3d(arg, -i, -j, -k);
      }
  
  return;
}

void _cpu_init_Greens_kernel_elements_micromag3d(float *dev_temp, int *Nkernel, int exchType, int *exchInConv, int co1, int co2, float *FD_cell_size, int *repetition, dev_qd_P_10, dev_qd_W_10){

  _cpu_init_Greens_kernel_elements_micromag3d_arg args;

  args.dev_temp = dev_temp;
  args.Nkernel = Nkernel;
  args.exchType = exchType;
  args.exchInConv = exchInConv;
  args.co1 = co1;
  args.co2 = co2;
  args.FD_cell_size = FD_cell_size;
  args.repetition = repetition;
  args.dev_qd_P_10 = dev_qd_P_10;
  args.dev_qd_W_10 = dev_qd_W_10;

  func_arg = (void *) (&args);

  thread_Wrapper(_cpu_init_Greens_kernel_elements_micromag3d_t);

  return;
}



typedef struct{
  float *dev_kernel_array, dev_temp;
  int N;
} _cpu_extract_real_parts_micromag3d_arg;

void _cpu_extract_real_parts_micromag3d_t(id){

  int start, stop;
  init_start_stop (&start, &stop, id, arg->N);

  for (int e=start; e<stop; e++)
    arg->dev_kernel_array[e] = arg->dev_temp[2*e];

  return;
}

void _cpu_extract_real_parts_micromag3d(float *dev_kernel_array, float *dev_temp, int N){

  _cpu_extract_real_parts_micromag3d_arg args;
  args.dev_kernel_array = dev_kernel_array;
  args.dev_temp = devtemp;
  args.N = N;

  func_arg = (void *) (&args);

  thread_Wrapper(_cpu_extract_real_parts_micromag3d_t);

  return;
}



void initialize_Gauss_quadrature_micromag3d(float *std_qd_W_10, float *mapped_qd_P_10, float *FD_cell_size){

  // initialize standard order 10 Gauss quadrature points and weights _____________________________
    float *std_qd_P_10 = (float*)calloc(10, sizeof(float));
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

    std_qd_W_10[0] = std_qd_W_10[9] = 0.066671344308687999f;
    std_qd_W_10[1] = std_qd_W_10[8] = 0.149451349150581f;
    std_qd_W_10[2] = std_qd_W_10[7] = 0.21908636251598201f;
    std_qd_W_10[3] = std_qd_W_10[6] = 0.26926671930999602f;
    std_qd_W_10[4] = std_qd_W_10[5] = 0.29552422471475298f;
  // ______________________________________________________________________________________________

  // Map the standard Gauss quadrature points to the used integration boundaries __________________
    get_Quad_Points_micromag3d(&mapped_qd_P_10[X*10], std_qd_P_10, 10, -0.5f*FD_cell_size[X], 0.5f*FD_cell_size[X]);
    get_Quad_Points_micromag3d(&mapped_qd_P_10[Y*10], std_qd_P_10, 10, -0.5f*FD_cell_size[Y], 0.5f*FD_cell_size[Y]);
    get_Quad_Points_micromag3d(&mapped_qd_P_10[Z*10], std_qd_P_10, 10, -0.5f*FD_cell_size[Z], 0.5f*FD_cell_size[Z]);
  // ______________________________________________________________________________________________


  free (std_qd_P_10);

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

