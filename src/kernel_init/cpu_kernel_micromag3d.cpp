#include "tensor.h"
#include <stdio.h>
#include "cpu_kernel_micromag3d.h"
#include "cpukern.h"
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif


void cpu_init_kernel_elements_micromag3d(int co1, int co2, int *kernelSize, float *cellSize, int *repetition){

  // initialization Gauss quadrature points for integrations ______________________________________
    float *qd_W_10 = (float*)calloc(10, sizeof(float));
    float *qd_P_10 = (float*)calloc(3*10, sizeof(float));
    initialize_Gauss_quadrature_micromag3d(qd_W_10, qd_P_10, cellSize);
  // ______________________________________________________________________________________________

  int kernelN = kernelSize[X]*kernelSize[Y]*kernelSize[Z];              // size of a kernel component without zeros
  float *data = new_cpu_array(kernelN);                                 // o store kernel component without zeros (input of fft routine)
 
  _cpu_init_kernel_elements_micromag3d(data, kernelSize, co1, co2, cellSize, repetition, qd_P_10, qd_W_10);

  free (qd_W_10);
  free (qd_P_10);
  
//   write_tensor_pieces(3, kernelSize, data, stdout);
  print_tensor(as_tensorN(data, 3, kernelSize));
  free (data);
  
  return;
}


typedef struct{
  float *cellSize, *data, *qd_P_10, *qd_W_10;
  int *Nkernel, *repetition; 
  int co1, co2;
} _cpu_init_kernel_elements_micromag3d_arg;

float _cpu_get_kernel_element_micromag3d(_cpu_init_kernel_elements_micromag3d_arg *arg, int a, int b, int c){

  int *Nkernel = arg->Nkernel;
  int co1 = arg->co1;
  int co2 = arg->co2;
  float *cellSize = arg->cellSize;
  int *repetition = arg->repetition;
  float *qd_P_10 = arg->qd_P_10;
  float *qd_W_10 = arg->qd_W_10;
  
  float result = 0.0f;
  float *qd_P_10_X = &qd_P_10[X*10];
  float *qd_P_10_Y = &qd_P_10[Y*10];
  float *qd_P_10_Z = &qd_P_10[Z*10];
  
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
          float x1 = (i + 0.5f) * cellSize[X];
          float x2 = (i - 0.5f) * cellSize[X];
          for (int cnt2=0; cnt2<10; cnt2++){
            float y = j * cellSize[Y] + qd_P_10_Y[cnt2];
            for (int cnt3=0; cnt3<10; cnt3++){
              float z = k * cellSize[Z] + qd_P_10_Z[cnt3];
              result += cellSize[Y] * cellSize[Z] / 4.0f * qd_W_10[cnt2] * qd_W_10[cnt3] *
                ( x1*powf(x1*x1+y*y+z*z, -1.5f) - x2*powf(x2*x2+y*y+z*z, -1.5f));
            }
          }
        }
        else{
          float r2 = (i*cellSize[X])*(i*cellSize[X]) + (j*cellSize[Y])*(j*cellSize[Y]) + (k*cellSize[Z])*(k*cellSize[Z]);
          result += cellSize[X] * cellSize[Y] * cellSize[Z] *
                    (1.0f/ powf(r2,1.5f) - 3.0f* (i*cellSize[X]) * (i*cellSize[X]) * powf(r2,-2.5f));
        }
        
      
/*        if (r2_int<400){   // The source is considered as a uniformly magnetized FD cell and the field is averaged over the complete observation FD cell
          for (int cnta=0; cnta<10; cnta++)
          for (int cntb=0; cntb<10; cntb++)
          for (int cntc=0; cntc<10; cntc++){
            float x1 = (i + 0.5f) * cellSize[X] + qd_P_10_X[cnta];
            float x2 = (i - 0.5f) * cellSize[X] + qd_P_10_X[cnta];
            for (int cnt2=0; cnt2<10; cnt2++){
              float y = j * cellSize[Y] + qd_P_10_Y[cnt2] + qd_P_10_Y[cntb];
              for (int cnt3=0; cnt3<10; cnt3++){
                float z = k * cellSize[Z] + qd_P_10_Z[cnt3] + qd_P_10_Z[cntc];
                result += 1.0f/8.0f * cellSize[Y] * cellSize[Z] / 4.0f * qd_W_10[cnt2] * qd_W_10[cnt3] *
                  qd_W_10[cnta] * qd_W_10[cntb] * qd_W_10[cntc] *
                  ( x1*powf(x1*x1+y*y+z*z, -1.5f) - x2*powf(x2*x2+y*y+z*z, -1.5f) );
              }
            }
          }
        }
        else{   // The source is considered as a point. No averaging of the field in the observation FD cell
          float r2 = (i*cellSize[X])*(i*cellSize[X]) + (j*cellSize[Y])*(j*cellSize[Y]) + (k*cellSize[Z])*(k*cellSize[Z]);
          result += cellSize[X] * cellSize[Y] * cellSize[Z] *
                    (1.0f/ powf(r2,1.5f) - 3.0f* (i*cellSize[X]) * (i*cellSize[X]) * powf(r2,-2.5f));
        }*/
        
      
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

        if (r2_int<0){
          float x1 = (i + 0.5f) * cellSize[X];
          float x2 = (i - 0.5f) * cellSize[X];
          for (int cnt2=0; cnt2<10; cnt2++){
            float y = j * cellSize[Y] + qd_P_10_Y[cnt2];
            for (int cnt3=0; cnt3<10; cnt3++){
              float z = k * cellSize[Z] + qd_P_10_Z[cnt3];
              result += 1.0f/8.0f * cellSize[Y] * cellSize[Z] / 4.0f * qd_W_10[cnt2] * qd_W_10[cnt3] *
                ( y*powf(x1*x1+y*y+z*z, -1.5f) - y*powf(x2*x2+y*y+z*z, -1.5f));
            }
          }
        }
        else{
          float r2 = (i*cellSize[X])*(i*cellSize[X]) + (j*cellSize[Y])*(j*cellSize[Y]) + (k*cellSize[Z])*(k*cellSize[Z]);
          result += cellSize[X] * cellSize[Y] * cellSize[Z] * 
                    (- 3.0f* (i*cellSize[X]) * (j*cellSize[Y]) * powf(r2,-2.5f));
        }
       
//         if (r2_int<400){
//           for (int cnta=0; cnta<10; cnta++)
//           for (int cntb=0; cntb<10; cntb++)
//           for (int cntc=0; cntc<10; cntc++){
//             float x1 = (i + 0.5f) * cellSize[X] + qd_P_10_X[cnta];
//             float x2 = (i - 0.5f) * cellSize[X] + qd_P_10_X[cnta];
//             for (int cnt2=0; cnt2<10; cnt2++){
//               float y = j * cellSize[Y] + qd_P_10_Y[cnt2] + qd_P_10_Y[cntb];
//               for (int cnt3=0; cnt3<10; cnt3++){
//                 float z = k * cellSize[Z] + qd_P_10_Z[cnt3] + qd_P_10_Z[cntc];
//                 result += 1.0f/8.0f * cellSize[Y] * cellSize[Z] / 4.0f * qd_W_10[cnt2] * qd_W_10[cnt3] *
//                   qd_W_10[cnta] * qd_W_10[cntb] * qd_W_10[cntc] *
//                   ( y*powf(x1*x1+y*y+z*z, -1.5f) - y*powf(x2*x2+y*y+z*z, -1.5f));
//               }
//             }
//           }
//         }
//         else{
//           float r2 = (i*cellSize[X])*(i*cellSize[X]) + (j*cellSize[Y])*(j*cellSize[Y]) + (k*cellSize[Z])*(k*cellSize[Z]);
//           result += cellSize[X] * cellSize[Y] * cellSize[Z] * 
//                     (- 3.0f* (i*cellSize[X]) * (j*cellSize[Y]) * powf(r2,-2.5f));
//         }

      }
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
          float x1 = (i + 0.5f) * cellSize[X];
          float x2 = (i - 0.5f) * cellSize[X];
          for (int cnt2=0; cnt2<10; cnt2++){
            float y = j * cellSize[Y] + qd_P_10_Y[cnt2];
            for (int cnt3=0; cnt3<10; cnt3++){
              float z = k * cellSize[Z] + qd_P_10_Z[cnt3];
              result += cellSize[Y] * cellSize[Z] / 4.0f * qd_W_10[cnt2] * qd_W_10[cnt3] *
                ( z*powf(x1*x1+y*y+z*z, -1.5f) - z*powf(x2*x2+y*y+z*z, -1.5f));
            }
          }
        }
        else{
          float r2 = (i*cellSize[X])*(i*cellSize[X]) + (j*cellSize[Y])*(j*cellSize[Y]) + (k*cellSize[Z])*(k*cellSize[Z]);
          result += cellSize[X] * cellSize[Y] * cellSize[Z] * 
                    (- 3.0f* (i*cellSize[X]) * (k*cellSize[Y]) * powf(r2,-2.5f));
        }

/*        if (r2_int<400){
          for (int cnta=0; cnta<10; cnta++)
          for (int cntb=0; cntb<10; cntb++)
          for (int cntc=0; cntc<10; cntc++){
            float x1 = (i + 0.5f) * cellSize[X] + qd_P_10_X[cnta];
            float x2 = (i - 0.5f) * cellSize[X] + qd_P_10_X[cnta];
            for (int cnt2=0; cnt2<10; cnt2++){
              float y = j * cellSize[Y] + qd_P_10_Y[cnt2] + qd_P_10_Y[cntb];
              for (int cnt3=0; cnt3<10; cnt3++){
                float z = k * cellSize[Z] + qd_P_10_Z[cnt3] + qd_P_10_Z[cntc];
                result += 1.0f/8.0f * cellSize[Y] * cellSize[Z] / 4.0f * qd_W_10[cnt2] * qd_W_10[cnt3] *
                  qd_W_10[cnta] * qd_W_10[cntb] * qd_W_10[cntc] *
                  ( z*powf(x1*x1+y*y+z*z, -1.5f) - z*powf(x2*x2+y*y+z*z, -1.5f));
              }
            }
          }
        }
        else{
          float r2 = (i*cellSize[X])*(i*cellSize[X]) + (j*cellSize[Y])*(j*cellSize[Y]) + (k*cellSize[Z])*(k*cellSize[Z]);
          result += cellSize[X] * cellSize[Y] * cellSize[Z] * 
                    (- 3.0f* (i*cellSize[X]) * (k*cellSize[Y]) * powf(r2,-2.5f));
        }*/
      
      
      }
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
          float y1 = (j + 0.5f) * cellSize[Y];
          float y2 = (j - 0.5f) * cellSize[Y];
          for (int cnt1=0; cnt1<10; cnt1++){
            float x = i * cellSize[X] + qd_P_10_X[cnt1];
            for (int cnt3=0; cnt3<10; cnt3++){
              float z = k * cellSize[Z] + qd_P_10_Z[cnt3];
              result += cellSize[X] * cellSize[Z] / 4.0f * qd_W_10[cnt1] * qd_W_10[cnt3] *
                ( y1*powf(x*x+y1*y1+z*z, -1.5f) - y2*powf(x*x+y2*y2+z*z, -1.5f));
            }
          }
        }
        else{
          float r2 = (i*cellSize[X])*(i*cellSize[X]) + (j*cellSize[Y])*(j*cellSize[Y]) + (k*cellSize[Z])*(k*cellSize[Z]);
          result += cellSize[X] * cellSize[Y] * cellSize[Z] * 
                    (1.0f/ powf(r2,1.5f) - 3.0f* (j*cellSize[Y]) * (j*cellSize[Y]) * powf(r2,-2.5f));
        }

      
        if (r2_int<400){
          for (int cnta=0; cnta<10; cnta++)
          for (int cntb=0; cntb<10; cntb++)
          for (int cntc=0; cntc<10; cntc++){
            float y1 = (j + 0.5f) * cellSize[Y] + qd_P_10_Y[cntb];
            float y2 = (j - 0.5f) * cellSize[Y] + qd_P_10_Y[cntb];
            for (int cnt1=0; cnt1<10; cnt1++){
              float x = i * cellSize[X] + qd_P_10_X[cnt1] + qd_P_10_X[cnta];
              for (int cnt3=0; cnt3<10; cnt3++){
                float z = k * cellSize[Z] + qd_P_10_Z[cnt3] + qd_P_10_Z[cntc];
                result += 1.0f/8.0f * cellSize[X] * cellSize[Z] / 4.0f * qd_W_10[cnt1] * qd_W_10[cnt3] *
                  qd_W_10[cnta] * qd_W_10[cntb] * qd_W_10[cntc] *
                  ( y1*powf(x*x+y1*y1+z*z, -1.5f) - y2*powf(x*x+y2*y2+z*z, -1.5f));
              }
            }
          }
        }
        else{
          float r2 = (i*cellSize[X])*(i*cellSize[X]) + (j*cellSize[Y])*(j*cellSize[Y]) + (k*cellSize[Z])*(k*cellSize[Z]);
          result += cellSize[X] * cellSize[Y] * cellSize[Z] * 
                    (1.0f/ powf(r2,1.5f) - 3.0f* (j*cellSize[Y]) * (j*cellSize[Y]) * powf(r2,-2.5f));
        }
      
      
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
          float y1 = (j + 0.5f) * cellSize[Y];
          float y2 = (j - 0.5f) * cellSize[Y];
          for (int cnt1=0; cnt1<10; cnt1++){
            float x = i * cellSize[X] + qd_P_10_X[cnt1];
            for (int cnt3=0; cnt3<10; cnt3++){
              float z = k * cellSize[Z] + qd_P_10_Z[cnt3];
              result += cellSize[X] * cellSize[Z] / 4.0f * qd_W_10[cnt1] * qd_W_10[cnt3] *
                ( z*powf(x*x+y1*y1+z*z, -1.5f) - z*powf(x*x+y2*y2+z*z, -1.5f));
            }
          }
        }
        else{
          float r2 = (i*cellSize[X])*(i*cellSize[X]) + (j*cellSize[Y])*(j*cellSize[Y]) + (k*cellSize[Z])*(k*cellSize[Z]);
          result += cellSize[X] * cellSize[Y] * cellSize[Z] * 
                    ( - 3.0f* (j*cellSize[Y]) * (k*cellSize[Z]) * powf(r2,-2.5f));
        }

/*        if (r2_int<400){
          for (int cnta=0; cnta<10; cnta++)
          for (int cntb=0; cntb<10; cntb++)
          for (int cntc=0; cntc<10; cntc++){
          float y1 = (j + 0.5f) * cellSize[Y] + qd_P_10_Y[cntb];
          float y2 = (j - 0.5f) * cellSize[Y] + qd_P_10_Y[cntb];
            for (int cnt1=0; cnt1<10; cnt1++){
              float x = i * cellSize[X] + qd_P_10_X[cnt1] + qd_P_10_X[cnta];
              for (int cnt3=0; cnt3<10; cnt3++){
                float z = k * cellSize[Z] + qd_P_10_Z[cnt3] + qd_P_10_Z[cntc];
                result += 1.0f/8.0f * cellSize[X] * cellSize[Z] / 4.0f * qd_W_10[cnt1] * qd_W_10[cnt3] *
                  qd_W_10[cnta] * qd_W_10[cntb] * qd_W_10[cntc] *
                  ( z*powf(x*x+y1*y1+z*z, -1.5f) - z*powf(x*x+y2*y2+z*z, -1.5f));
              }
            }
          }
        }
        else{
          float r2 = (i*cellSize[X])*(i*cellSize[X]) + (j*cellSize[Y])*(j*cellSize[Y]) + (k*cellSize[Z])*(k*cellSize[Z]);
          result += cellSize[X] * cellSize[Y] * cellSize[Z] * 
                    ( - 3.0f* (j*cellSize[Y]) * (k*cellSize[Z]) * powf(r2,-2.5f));
        }*/
      
      
      }
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
          float z1 = (k + 0.5f) * cellSize[Z];
          float z2 = (k - 0.5f) * cellSize[Z];
          for (int cnt1=0; cnt1<10; cnt1++){
            float x = i * cellSize[X] + qd_P_10_X[cnt1];
            for (int cnt2=0; cnt2<10; cnt2++){
              float y = j * cellSize[Y] + qd_P_10_Y[cnt2];
              result += cellSize[X] * cellSize[Y] / 4.0f * qd_W_10[cnt1] * qd_W_10[cnt2] *
                ( z1*powf(x*x+y*y+z1*z1, -1.5f) - z2*powf(x*x+y*y+z2*z2, -1.5f));
            }
          }
        }
        else{
          float r2 = (i*cellSize[X])*(i*cellSize[X]) + (j*cellSize[Y])*(j*cellSize[Y]) + (k*cellSize[Z])*(k*cellSize[Z]);
          result += cellSize[X] * cellSize[Y] * cellSize[Z] * 
                    (1.0f/ powf(r2,1.5f) - 3.0f* (k*cellSize[Z]) * (k*cellSize[Z]) * powf(r2,-2.5f));
        }

/*        if (r2_int<400){
          for (int cnta=0; cnta<10; cnta++)
          for (int cntb=0; cntb<10; cntb++)
          for (int cntc=0; cntc<10; cntc++){
          float z1 = (k + 0.5f) * cellSize[Z] + qd_P_10_Z[cntc];
          float z2 = (k - 0.5f) * cellSize[Z] + qd_P_10_Z[cntc];
            for (int cnt1=0; cnt1<10; cnt1++){
              float x = i * cellSize[X] + qd_P_10_X[cnt1] + qd_P_10_X[cnta];
              for (int cnt2=0; cnt2<10; cnt2++){
                float y = j * cellSize[Y] + qd_P_10_Y[cnt2] + qd_P_10_Y[cntb];
                result += 1.0f/8.0f * cellSize[X] * cellSize[Y] / 4.0f * qd_W_10[cnt1] * qd_W_10[cnt2] *
                  qd_W_10[cnta] * qd_W_10[cntb] * qd_W_10[cntc] *
                  ( z1*powf(x*x+y*y+z1*z1, -1.5f) - z2*powf(x*x+y*y+z2*z2, -1.5f));
              }
            }
          }
        }
        else{
          float r2 = (i*cellSize[X])*(i*cellSize[X]) + (j*cellSize[Y])*(j*cellSize[Y]) + (k*cellSize[Z])*(k*cellSize[Z]);
          result += cellSize[X] * cellSize[Y] * cellSize[Z] * 
                    (1.0f/ powf(r2,1.5f) - 3.0f* (k*cellSize[Z]) * (k*cellSize[Z]) * powf(r2,-2.5f));
        }*/
      }
    }
  // ______________________________________________________________________________________________
  
  result *= -1.0f/4.0f/3.14159265f;
  return( result );
}

void _cpu_init_kernel_elements_micromag3d_t(int id){
  
  _cpu_init_kernel_elements_micromag3d_arg *arg = (_cpu_init_kernel_elements_micromag3d_arg *) func_arg;

  int start, stop;
  init_start_stop (&start, &stop, id, (arg->Nkernel[X]+1)/2);

  int N2 = arg->Nkernel[Z];
  int N12 = arg->Nkernel[Y] * N2;
  

  for (int i=start; i<stop; i++)
    for (int j=0; j<arg->Nkernel[Y]/2; j++)
      for (int k=0; k<N2/2; k++){
          arg->data[                  i*N12 +                   j*N2 +                 k] = _cpu_get_kernel_element_micromag3d(arg, i, j, k);
        if (i>0)
          arg->data[(arg->Nkernel[X]-i)*N12 +                   j*N2 +                 k] = _cpu_get_kernel_element_micromag3d(arg, -i, j, k);
        if (j>0)
          arg->data[                  i*N12 + (arg->Nkernel[Y]-j)*N2 +                 k] = _cpu_get_kernel_element_micromag3d(arg, i, -j, k);
        if (k>0) 
          arg->data[                  i*N12 +                   j*N2 + arg->Nkernel[Z]-k] = _cpu_get_kernel_element_micromag3d(arg, i, j, -k);
        if (i>0 && j>0)
          arg->data[(arg->Nkernel[X]-i)*N12 + (arg->Nkernel[Y]-j)*N2 +                 k] = _cpu_get_kernel_element_micromag3d(arg, -i, -j, k);
        if (i>0 && k>0) 
          arg->data[(arg->Nkernel[X]-i)*N12 +                   j*N2 + arg->Nkernel[Z]-k] = _cpu_get_kernel_element_micromag3d(arg, -i, j, -k);
        if (j>0 && k>0) 
          arg->data[                  i*N12 + (arg->Nkernel[Y]-j)*N2 + arg->Nkernel[Z]-k] = _cpu_get_kernel_element_micromag3d(arg, i, -j, -k);
        if (i>0 && j>0 && k>0) 
          arg->data[(arg->Nkernel[X]-i)*N12 + (arg->Nkernel[Y]-j)*N2 + arg->Nkernel[Z]-k] = _cpu_get_kernel_element_micromag3d(arg, -i, -j, -k);
      }
  
  return;
}

void _cpu_init_kernel_elements_micromag3d(float *data, int *Nkernel, int co1, int co2, float *cellSize, int *repetition, float *qd_P_10, float *qd_W_10){

  _cpu_init_kernel_elements_micromag3d_arg args;

  args.data = data;
  args.Nkernel = Nkernel;
  args.co1 = co1;
  args.co2 = co2;
  args.cellSize = cellSize;
  args.repetition = repetition;
  args.qd_P_10 = qd_P_10;
  args.qd_W_10 = qd_W_10;

  func_arg = (void *) (&args);

  thread_Wrapper(_cpu_init_kernel_elements_micromag3d_t);

  return;
}


void initialize_Gauss_quadrature_micromag3d(float *std_qd_W_10, float *mapped_qd_P_10, float *cellSize){

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
    get_Quad_Points_micromag3d(&mapped_qd_P_10[X*10], std_qd_P_10, 10, -0.5f*cellSize[X], 0.5f*cellSize[X]);
    get_Quad_Points_micromag3d(&mapped_qd_P_10[Y*10], std_qd_P_10, 10, -0.5f*cellSize[Y], 0.5f*cellSize[Y]);
    get_Quad_Points_micromag3d(&mapped_qd_P_10[Z*10], std_qd_P_10, 10, -0.5f*cellSize[Z], 0.5f*cellSize[Z]);
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

