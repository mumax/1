#include "tensor.h"
#include <stdio.h>
#include "gpu_kernel_micromag3d.h"
#include "gpukern.h"
#include "cpu_mem.h"

#ifdef __cplusplus
extern "C" {
#endif


__device__ float _gpu_get_kernel_element_micromag3d(int Nkernel_X, int Nkernel_Y, int Nkernel_Z, int co1, int co2, int a, int b, int c, float cellSize_X, float cellSize_Y, float cellSize_Z, int repetition_X, int repetition_Y, int repetition_Z, float *dev_qd_P_10, float *dev_qd_W_10){

  float result = 0.0f;
  float *dev_qd_P_10_X = &dev_qd_P_10[X*10];
  float *dev_qd_P_10_Y = &dev_qd_P_10[Y*10];
  float *dev_qd_P_10_Z = &dev_qd_P_10[Z*10];
  
  // for elements in Kernel component gxx _________________________________________________________
    if (co1==0 && co2==0){

      for(int cnta=-repetition_X; cnta<=repetition_X; cnta++)
      for(int cntb=-repetition_Y; cntb<=repetition_Y; cntb++)
      for(int cntc=-repetition_Z; cntc<=repetition_Z; cntc++){

        int i = a + cnta*Nkernel_X/2;
        int j = b + cntb*Nkernel_Y/2;
        int k = c + cntc*Nkernel_Z/2;
        int r2_int = i*i+j*j+k*k;

/*        if (r2_int<400){
          float x1 = (i + 0.5f) * cellSize_X;
          float x2 = (i - 0.5f) * cellSize_X;
          for (int cnt2=0; cnt2<10; cnt2++){
            float y = j * cellSize_Y + dev_qd_P_10_Y[cnt2];
            for (int cnt3=0; cnt3<10; cnt3++){
              float z = k * cellSize_Z + dev_qd_P_10_Z[cnt3];
              result += cellSize_Y * cellSize_Z / 4.0f * dev_qd_W_10[cnt2] * dev_qd_W_10[cnt3] *
                ( x1*__powf(x1*x1+y*y+z*z, -1.5f) - x2*__powf(x2*x2+y*y+z*z, -1.5f));
            }
          }
        }
        else{
          float r2 = (i*cellSize_X)*(i*cellSize_X) + (j*cellSize_Y)*(j*cellSize_Y) + (k*cellSize_Z)*(k*cellSize_Z);
          result += cellSize_X * cellSize_Y * cellSize_Z *
                    (1.0f/ __powf(r2,1.5f) - 3.0f* (i*cellSize_X) * (i*cellSize_X) * __powf(r2,-2.5f));
        }*/
        
        if (r2_int<400){   // The source is considered as a uniformly magnetized FD cell and the field is averaged over the complete observation FD cell
          for (int cnta=0; cnta<10; cnta++)
          for (int cntb=0; cntb<10; cntb++)
          for (int cntc=0; cntc<10; cntc++){
            float x1 = (i + 0.5f) * cellSize_X + dev_qd_P_10_X[cnta];
            float x2 = (i - 0.5f) * cellSize_X + dev_qd_P_10_X[cnta];
            for (int cnt2=0; cnt2<10; cnt2++){
              float y = j * cellSize_Y + dev_qd_P_10_Y[cnt2] + dev_qd_P_10_Y[cntb];
              for (int cnt3=0; cnt3<10; cnt3++){
                float z = k * cellSize_Z + dev_qd_P_10_Z[cnt3] + dev_qd_P_10_Z[cntc];
                result += 1.0f/8.0f * cellSize_Y * cellSize_Z / 4.0f * dev_qd_W_10[cnt2] * dev_qd_W_10[cnt3] *
                  ( x1*__powf(x1*x1+y*y+z*z, -1.5f) - x2*__powf(x2*x2+y*y+z*z, -1.5f));
              }
            }
          }
        }
        else{   // The source is considered as a point. No averaging of the field in the observation FD cell
          float r2 = (i*cellSize_X)*(i*cellSize_X) + (j*cellSize_Y)*(j*cellSize_Y) + (k*cellSize_Z)*(k*cellSize_Z);
          result += cellSize_X * cellSize_Y * cellSize_Z *
                    (1.0f/ __powf(r2,1.5f) - 3.0f* (i*cellSize_X) * (i*cellSize_X) * __powf(r2,-2.5f));
        }
        
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

//         if (r2_int<400){
//           float x1 = (i + 0.5f) * cellSize_X;
//           float x2 = (i - 0.5f) * cellSize_X;
//           for (int cnt2=0; cnt2<10; cnt2++){
//             float y = j * cellSize_Y + dev_qd_P_10_Y[cnt2];
//             for (int cnt3=0; cnt3<10; cnt3++){
//               float z = k * cellSize_Z + dev_qd_P_10_Z[cnt3];
//               result += cellSize_Y * cellSize_Z / 4.0f * dev_qd_W_10[cnt2] * dev_qd_W_10[cnt3] *
//                 ( y*__powf(x1*x1+y*y+z*z, -1.5f) - y*__powf(x2*x2+y*y+z*z, -1.5f));
//             }
//           }
//         }
//         else{
//           float r2 = (i*cellSize_X)*(i*cellSize_X) + (j*cellSize_Y)*(j*cellSize_Y) + (k*cellSize_Z)*(k*cellSize_Z);
//           result += cellSize_X * cellSize_Y * cellSize_Z * 
//                     (- 3.0f* (i*cellSize_X) * (j*cellSize_Y) * __powf(r2,-2.5f));
//         }

        if (r2_int<400){
          for (int cnta=0; cnta<10; cnta++)
          for (int cntb=0; cntb<10; cntb++)
          for (int cntc=0; cntc<10; cntc++){
            float x1 = (i + 0.5f) * cellSize_X + dev_qd_P_10_X[cnta];
            float x2 = (i - 0.5f) * cellSize_X + dev_qd_P_10_X[cnta];
            for (int cnt2=0; cnt2<10; cnt2++){
              float y = j * cellSize_Y + dev_qd_P_10_Y[cnt2] + dev_qd_P_10_Y[cntb];
              for (int cnt3=0; cnt3<10; cnt3++){
                float z = k * cellSize_Z + dev_qd_P_10_Z[cnt3] + dev_qd_P_10_Z[cntc];
                result += 1.0f/8.0f * cellSize_Y * cellSize_Z / 4.0f * dev_qd_W_10[cnt2] * dev_qd_W_10[cnt3] *
                  ( y*__powf(x1*x1+y*y+z*z, -1.5f) - y*__powf(x2*x2+y*y+z*z, -1.5f));
              }
            }
          }
        }
        else{
          float r2 = (i*cellSize_X)*(i*cellSize_X) + (j*cellSize_Y)*(j*cellSize_Y) + (k*cellSize_Z)*(k*cellSize_Z);
          result += cellSize_X * cellSize_Y * cellSize_Z * 
                    (- 3.0f* (i*cellSize_X) * (j*cellSize_Y) * __powf(r2,-2.5f));
        }
        
        
      }
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

/*        if (r2_int<400){
          float x1 = (i + 0.5f) * cellSize_X;
          float x2 = (i - 0.5f) * cellSize_X;
          for (int cnt2=0; cnt2<10; cnt2++){
            float y = j * cellSize_Y + dev_qd_P_10_Y[cnt2];
            for (int cnt3=0; cnt3<10; cnt3++){
              float z = k * cellSize_Z + dev_qd_P_10_Z[cnt3];
              result += cellSize_Y * cellSize_Z / 4.0f * dev_qd_W_10[cnt2] * dev_qd_W_10[cnt3] *
                ( z*__powf(x1*x1+y*y+z*z, -1.5f) - z*__powf(x2*x2+y*y+z*z, -1.5f));
            }
          }
        }
        else{
          float r2 = (i*cellSize_X)*(i*cellSize_X) + (j*cellSize_Y)*(j*cellSize_Y) + (k*cellSize_Z)*(k*cellSize_Z);
          result += cellSize_X * cellSize_Y * cellSize_Z * 
                    (- 3.0f* (i*cellSize_X) * (k*cellSize_Y) * __powf(r2,-2.5f));
        }*/
        
        if (r2_int<400){
          for (int cnta=0; cnta<10; cnta++)
          for (int cntb=0; cntb<10; cntb++)
          for (int cntc=0; cntc<10; cntc++){
            float x1 = (i + 0.5f) * cellSize_X + dev_qd_P_10_X[cnta];
            float x2 = (i - 0.5f) * cellSize_X + dev_qd_P_10_X[cnta];
            for (int cnt2=0; cnt2<10; cnt2++){
              float y = j * cellSize_Y + dev_qd_P_10_Y[cnt2] + dev_qd_P_10_Y[cntb];
              for (int cnt3=0; cnt3<10; cnt3++){
                float z = k * cellSize_Z + dev_qd_P_10_Z[cnt3] + dev_qd_P_10_Z[cntc];
                result += 1.0f/8.0f * cellSize_Y * cellSize_Z / 4.0f * dev_qd_W_10[cnt2] * dev_qd_W_10[cnt3] *
                  ( z*__powf(x1*x1+y*y+z*z, -1.5f) - z*__powf(x2*x2+y*y+z*z, -1.5f));
              }
            }
          }
        }
        else{
          float r2 = (i*cellSize_X)*(i*cellSize_X) + (j*cellSize_Y)*(j*cellSize_Y) + (k*cellSize_Z)*(k*cellSize_Z);
          result += cellSize_X * cellSize_Y * cellSize_Z * 
                    (- 3.0f* (i*cellSize_X) * (k*cellSize_Y) * __powf(r2,-2.5f));
        }
        
      }
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

/*        if (r2_int<400){
          float y1 = (j + 0.5f) * cellSize_Y;
          float y2 = (j - 0.5f) * cellSize_Y;
          for (int cnt1=0; cnt1<10; cnt1++){
            float x = i * cellSize_X + dev_qd_P_10_X[cnt1];
            for (int cnt3=0; cnt3<10; cnt3++){
              float z = k * cellSize_Z + dev_qd_P_10_Z[cnt3];
              result += cellSize_X * cellSize_Z / 4.0f * dev_qd_W_10[cnt1] * dev_qd_W_10[cnt3] *
                ( y1*__powf(x*x+y1*y1+z*z, -1.5f) - y2*__powf(x*x+y2*y2+z*z, -1.5f));
            }
          }
        }
        else{
          float r2 = (i*cellSize_X)*(i*cellSize_X) + (j*cellSize_Y)*(j*cellSize_Y) + (k*cellSize_Z)*(k*cellSize_Z);
          result += cellSize_X * cellSize_Y * cellSize_Z * 
                    (1.0f/ __powf(r2,1.5f) - 3.0f* (j*cellSize_Y) * (j*cellSize_Y) * __powf(r2,-2.5f));
        }*/
        
        if (r2_int<400){
          for (int cnta=0; cnta<10; cnta++)
          for (int cntb=0; cntb<10; cntb++)
          for (int cntc=0; cntc<10; cntc++){
            float y1 = (j + 0.5f) * cellSize_Y + dev_qd_P_10_Y[cntb];
            float y2 = (j - 0.5f) * cellSize_Y + dev_qd_P_10_Y[cntb];
            for (int cnt1=0; cnt1<10; cnt1++){
              float x = i * cellSize_X + dev_qd_P_10_X[cnt1] + dev_qd_P_10_X[cnta];
              for (int cnt3=0; cnt3<10; cnt3++){
                float z = k * cellSize_Z + dev_qd_P_10_Z[cnt3] + dev_qd_P_10_Z[cntc];
                result += 1.0f/8.0f * cellSize_X * cellSize_Z / 4.0f * dev_qd_W_10[cnt1] * dev_qd_W_10[cnt3] *
                  ( y1*__powf(x*x+y1*y1+z*z, -1.5f) - y2*__powf(x*x+y2*y2+z*z, -1.5f));
              }
            }
          }
        }
        else{
          float r2 = (i*cellSize_X)*(i*cellSize_X) + (j*cellSize_Y)*(j*cellSize_Y) + (k*cellSize_Z)*(k*cellSize_Z);
          result += cellSize_X * cellSize_Y * cellSize_Z * 
                    (1.0f/ __powf(r2,1.5f) - 3.0f* (j*cellSize_Y) * (j*cellSize_Y) * __powf(r2,-2.5f));
        }
        
        
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

/*        if (r2_int<400){
          float y1 = (j + 0.5f) * cellSize_Y;
          float y2 = (j - 0.5f) * cellSize_Y;
          for (int cnt1=0; cnt1<10; cnt1++){
            float x = i * cellSize_X + dev_qd_P_10_X[cnt1];
            for (int cnt3=0; cnt3<10; cnt3++){
              float z = k * cellSize_Z + dev_qd_P_10_Z[cnt3];
              result += cellSize_X * cellSize_Z / 4.0f * dev_qd_W_10[cnt1] * dev_qd_W_10[cnt3] *
                ( z*__powf(x*x+y1*y1+z*z, -1.5f) - z*__powf(x*x+y2*y2+z*z, -1.5f));
            }
          }
        }
        else{
          float r2 = (i*cellSize_X)*(i*cellSize_X) + (j*cellSize_Y)*(j*cellSize_Y) + (k*cellSize_Z)*(k*cellSize_Z);
          result += cellSize_X * cellSize_Y * cellSize_Z * 
                    ( - 3.0f* (j*cellSize_Y) * (k*cellSize_Z) * __powf(r2,-2.5f));
        }*/
        
        if (r2_int<400){
          for (int cnta=0; cnta<10; cnta++)
          for (int cntb=0; cntb<10; cntb++)
          for (int cntc=0; cntc<10; cntc++){
          float y1 = (j + 0.5f) * cellSize_Y + dev_qd_P_10_Y[cntb];
          float y2 = (j - 0.5f) * cellSize_Y + dev_qd_P_10_Y[cntb];
            for (int cnt1=0; cnt1<10; cnt1++){
              float x = i * cellSize_X + dev_qd_P_10_X[cnt1] + dev_qd_P_10_X[cnta];
              for (int cnt3=0; cnt3<10; cnt3++){
                float z = k * cellSize_Z + dev_qd_P_10_Z[cnt3] + dev_qd_P_10_Z[cntc];
                result += 1.0f/8.0f * cellSize_X * cellSize_Z / 4.0f * dev_qd_W_10[cnt1] * dev_qd_W_10[cnt3] *
                  ( z*__powf(x*x+y1*y1+z*z, -1.5f) - z*__powf(x*x+y2*y2+z*z, -1.5f));
              }
            }
          }
        }
        else{
          float r2 = (i*cellSize_X)*(i*cellSize_X) + (j*cellSize_Y)*(j*cellSize_Y) + (k*cellSize_Z)*(k*cellSize_Z);
          result += cellSize_X * cellSize_Y * cellSize_Z * 
                    ( - 3.0f* (j*cellSize_Y) * (k*cellSize_Z) * __powf(r2,-2.5f));
        }
        
        
      }
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

/*        if (r2_int<400){
          float z1 = (k + 0.5f) * cellSize_Z;
          float z2 = (k - 0.5f) * cellSize_Z;
          for (int cnt1=0; cnt1<10; cnt1++){
            float x = i * cellSize_X + dev_qd_P_10_X[cnt1];
            for (int cnt2=0; cnt2<10; cnt2++){
              float y = j * cellSize_Y + dev_qd_P_10_Y[cnt2];
              result += cellSize_X * cellSize_Y / 4.0f * dev_qd_W_10[cnt1] * dev_qd_W_10[cnt2] *
                ( z1*__powf(x*x+y*y+z1*z1, -1.5f) - z2*__powf(x*x+y*y+z2*z2, -1.5f));
            }
          }
        }
        else{
          float r2 = (i*cellSize_X)*(i*cellSize_X) + (j*cellSize_Y)*(j*cellSize_Y) + (k*cellSize_Z)*(k*cellSize_Z);
          result += cellSize_X * cellSize_Y * cellSize_Z * 
                    (1.0f/ __powf(r2,1.5f) - 3.0f* (k*cellSize_Z) * (k*cellSize_Z) * __powf(r2,-2.5f));
        }*/
       
        if (r2_int<400){
          for (int cnta=0; cnta<10; cnta++)
          for (int cntb=0; cntb<10; cntb++)
          for (int cntc=0; cntc<10; cntc++){
          float z1 = (k + 0.5f) * cellSize_Z + dev_qd_P_10_Z[cntc];
          float z2 = (k - 0.5f) * cellSize_Z + dev_qd_P_10_Z[cntc];
            for (int cnt1=0; cnt1<10; cnt1++){
              float x = i * cellSize_X + dev_qd_P_10_X[cnt1] + dev_qd_P_10_X[cnta];
              for (int cnt2=0; cnt2<10; cnt2++){
                float y = j * cellSize_Y + dev_qd_P_10_Y[cnt2] + dev_qd_P_10_Y[cntb];
                result += 1.0f/8.0f * cellSize_X * cellSize_Y / 4.0f * dev_qd_W_10[cnt1] * dev_qd_W_10[cnt2] *
                  ( z1*__powf(x*x+y*y+z1*z1, -1.5f) - z2*__powf(x*x+y*y+z2*z2, -1.5f));
              }
            }
          }
        }
        else{
          float r2 = (i*cellSize_X)*(i*cellSize_X) + (j*cellSize_Y)*(j*cellSize_Y) + (k*cellSize_Z)*(k*cellSize_Z);
          result += cellSize_X * cellSize_Y * cellSize_Z * 
                    (1.0f/ powf(r2,1.5f) - 3.0f* (k*cellSize_Z) * (k*cellSize_Z) * __powf(r2,-2.5f));
        }
       
      }
    }
  // ______________________________________________________________________________________________
  
  result *= -1.0f/4.0f/3.14159265f;
  return( result );
}


__global__ void _gpu_init_kernel_elements_micromag3d(float *data, int Nkernel_X, int Nkernel_Y, int Nkernel_Z, int co1, int co2, float cellSize_X, float cellSize_Y, float cellSize_Z, int repetition_X, int repetition_Y, int repetition_Z, float *dev_qd_P_10, float *dev_qd_W_10){

  int i = blockIdx.x;
  int j = blockIdx.y;

  int N2 = Nkernel_Z;
  int N12 = Nkernel_Y * N2;

  if ( i<((Nkernel_X+1)/2) && j<(Nkernel_Y/2) )
    for (int k=0; k<N2/2; k++){
        data[            i*N12 +             j*N2 +           k] = _gpu_get_kernel_element_micromag3d(Nkernel_X, Nkernel_Y, Nkernel_Z, co1, co2,  i,  j,  k, cellSize_X, cellSize_Y, cellSize_Z, repetition_X, repetition_Y, repetition_Z, dev_qd_P_10, dev_qd_W_10);
      if (i>0)
        data[(Nkernel_X-i)*N12 +             j*N2 +           k] = _gpu_get_kernel_element_micromag3d(Nkernel_X, Nkernel_Y, Nkernel_Z, co1, co2, -i,  j,  k, cellSize_X, cellSize_Y, cellSize_Z, repetition_X, repetition_Y, repetition_Z, dev_qd_P_10, dev_qd_W_10);
      if (j>0)
        data[            i*N12 + (Nkernel_Y-j)*N2 +           k] = _gpu_get_kernel_element_micromag3d(Nkernel_X, Nkernel_Y, Nkernel_Z, co1, co2,  i, -j,  k, cellSize_X, cellSize_Y, cellSize_Z, repetition_X, repetition_Y, repetition_Z, dev_qd_P_10, dev_qd_W_10);
      if (k>0) 
        data[            i*N12 +             j*N2 + Nkernel_Z-k] = _gpu_get_kernel_element_micromag3d(Nkernel_X, Nkernel_Y, Nkernel_Z, co1, co2,  i,  j, -k, cellSize_X, cellSize_Y, cellSize_Z, repetition_X, repetition_Y, repetition_Z, dev_qd_P_10, dev_qd_W_10);
      if (i>0 && j>0)
        data[(Nkernel_X-i)*N12 + (Nkernel_Y-j)*N2 +           k] = _gpu_get_kernel_element_micromag3d(Nkernel_X, Nkernel_Y, Nkernel_Z, co1, co2, -i, -j,  k, cellSize_X, cellSize_Y, cellSize_Z, repetition_X, repetition_Y, repetition_Z, dev_qd_P_10, dev_qd_W_10);
      if (i>0 && k>0) 
        data[(Nkernel_X-i)*N12 +             j*N2 + Nkernel_Z-k] = _gpu_get_kernel_element_micromag3d(Nkernel_X, Nkernel_Y, Nkernel_Z, co1, co2, -i,  j, -k, cellSize_X, cellSize_Y, cellSize_Z, repetition_X, repetition_Y, repetition_Z, dev_qd_P_10, dev_qd_W_10);
      if (j>0 && k>0) 
        data[            i*N12 + (Nkernel_Y-j)*N2 + Nkernel_Z-k] = _gpu_get_kernel_element_micromag3d(Nkernel_X, Nkernel_Y, Nkernel_Z, co1, co2,  i, -j, -k, cellSize_X, cellSize_Y, cellSize_Z, repetition_X, repetition_Y, repetition_Z, dev_qd_P_10, dev_qd_W_10);
      if (i>0 && j>0 && k>0) 
        data[(Nkernel_X-i)*N12 + (Nkernel_Y-j)*N2 + Nkernel_Z-k] = _gpu_get_kernel_element_micromag3d(Nkernel_X, Nkernel_Y, Nkernel_Z, co1, co2, -i, -j, -k, cellSize_X, cellSize_Y, cellSize_Z, repetition_X, repetition_Y, repetition_Z, dev_qd_P_10, dev_qd_W_10);
    }
  
  return;
}


void gpu_init_kernel_elements_micromag3d(int co1, int co2, int *kernelSize, float *cellSize, int *repetition){

  // initialization Gauss quadrature points for integrations + copy to gpu ________________________
    float *dev_qd_W_10 = new_gpu_array(10);
    float *dev_qd_P_10 = new_gpu_array(3*10);
    initialize_Gauss_quadrature_on_gpu_micromag3d(dev_qd_W_10, dev_qd_P_10, cellSize);
  // ______________________________________________________________________________________________
  
  int kernelN = kernelSize[X]*kernelSize[Y]*kernelSize[Z];     // size of a kernel component without zeros
  float *data = new_gpu_array(kernelN);                        // to store kernel component without zeros (input of fft routine)
  
  dim3 gridsize((kernelSize[X]+1)/2, kernelSize[Y]/2, 1);
  dim3 blocksize(1,1,1);
  check3dconf(gridsize, blocksize);

  _gpu_init_kernel_elements_micromag3d<<<gridsize, blocksize>>>(data, kernelSize[X], kernelSize[Y], kernelSize[Z], co1, co2, cellSize[X], cellSize[Y], cellSize[Z], repetition[X], repetition[Y], repetition[Z], dev_qd_P_10, dev_qd_W_10);
  gpu_sync();

  cudaFree (dev_qd_W_10);
  cudaFree (dev_qd_P_10);

  float *cpu_data = (float *) calloc(kernelN, sizeof(float));
  memcpy_from_gpu(data, cpu_data, kernelN);

  //Arne: copy to local memory
  float* localdata = new_cpu_array(kernelN);
  memcpy_from_gpu(data, localdata, kernelN);
//   print_tensor(as_tensorN(localdata, 3, kernelSize));
  write_tensor_pieces(3, kernelSize, localdata, stdout);
  free_gpu_array (data);
  free_cpu_array(localdata);	
  return;
}



void initialize_Gauss_quadrature_on_gpu_micromag3d(float *dev_qd_W_10, float *dev_qd_P_10, float *cellSize){

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
    get_Quad_Points_micromag3d(&host_qd_P_10[X*10], std_qd_P_10, 10, -0.5f*cellSize[X], 0.5f*cellSize[X]);
    get_Quad_Points_micromag3d(&host_qd_P_10[Y*10], std_qd_P_10, 10, -0.5f*cellSize[Y], 0.5f*cellSize[Y]);
    get_Quad_Points_micromag3d(&host_qd_P_10[Z*10], std_qd_P_10, 10, -0.5f*cellSize[Z], 0.5f*cellSize[Z]);
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

