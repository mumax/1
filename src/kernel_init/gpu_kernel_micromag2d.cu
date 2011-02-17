#include "tensor.h"
#include <stdio.h>
#include "gpu_kernel_micromag2d.h"
#include "gpukern.h"

#ifdef __cplusplus
extern "C" {
#endif

__device__ float _gpu_get_kernel_element_micromag2d(int Nkernel_Y, int Nkernel_Z, int co1, int co2, int b, int c, float cellSize_Y, float cellSize_Z, int repetition_Y, int repetition_Z, float *dev_qd_P_10, float *dev_qd_W_10){

  float result = 0.0f;
  float *dev_qd_P_10_Y = &dev_qd_P_10[ 0];
  float *dev_qd_P_10_Z = &dev_qd_P_10[10];
  

  // for elements in Kernel component gyy _________________________________________________________
    if (co1==1 && co2==1){
      for(int cntb=-repetition_Y; cntb<=repetition_Y; cntb++)
      for(int cntc=-repetition_Z; cntc<=repetition_Z; cntc++){

        int j = b + cntb*Nkernel_Y/2;
        int k = c + cntc*Nkernel_Z/2;
        int r2_int = j*j+k*k;

        if (r2_int<400){
          for (int cnt2=0; cnt2<10; cnt2++){
            float y = j * cellSize_Y + dev_qd_P_10_Y[cnt2];
            for (int cnt3=0; cnt3<10; cnt3++){
              float z = k * cellSize_Z + dev_qd_P_10_Z[cnt3];
              result += cellSize_Y * cellSize_Z / 4.0f * dev_qd_W_10[cnt2] * dev_qd_W_10[cnt3] *
                ( 1.0f/(y*y+z*z) - 2.0f*y*y/(y*y+z*z)/(y*y+z*z) );
            }
          }
        }
        else{
          float r2 = (j*cellSize_Y)*(j*cellSize_Y) + (k*cellSize_Z)*(k*cellSize_Z);
          result += cellSize_Y * cellSize_Z * 
                    (1.0f/ r2 - 2.0f* (j*cellSize_Y) * (j*cellSize_Y)/r2/r2 );
        }

/*        if (r2_int<400){
          for (int cntb=0; cntb<10; cntb++)
          for (int cntc=0; cntc<10; cntc++){
            for (int cnt2=0; cnt2<10; cnt2++){
              float y = j * cellSize_Y + dev_qd_P_10_Y[cnt2] + dev_qd_P_10_Y[cntb];
              for (int cnt3=0; cnt3<10; cnt3++){
                float z = k * cellSize_Z + dev_qd_P_10_Z[cnt3] + dev_qd_P_10_Z[cntc];
                result += 1.0f/4.0f * cellSize_Y * cellSize_Z / 4.0f * dev_qd_W_10[cnt2] * dev_qd_W_10[cnt3] *
                  ( 1.0f/(y*y+z*z) - 2.0f*y*y/(y*y+z*z)/(y*y+z*z) );
              }
            }
          }
        }
        else{
          float r2 = (j*cellSize_Y)*(j*cellSize_Y) + (k*cellSize_Z)*(k*cellSize_Z);
          result += cellSize_Y * cellSize_Z * 
                    (1.0f/ r2 - 2.0f* (j*cellSize_Y) * (j*cellSize_Y)/r2/r2 );
        }*/
        
      }
    }
  // ______________________________________________________________________________________________


  // for elements in Kernel component gyz _________________________________________________________
    if (co1==1 && co2==2){
      for(int cntb=-repetition_Y; cntb<=repetition_Y; cntb++)
      for(int cntc=-repetition_Z; cntc<=repetition_Z; cntc++){

        int j = b + cntb*Nkernel_Y/2;
        int k = c + cntc*Nkernel_Z/2;
        int r2_int = j*j+k*k;

        if (r2_int<400){
          for (int cnt2=0; cnt2<10; cnt2++){
            float y = j * cellSize_Y + dev_qd_P_10_Y[cnt2];
            for (int cnt3=0; cnt3<10; cnt3++){
              float z = k * cellSize_Z + dev_qd_P_10_Z[cnt3];
              result += cellSize_Y * cellSize_Z / 4.0f * dev_qd_W_10[cnt2] * dev_qd_W_10[cnt3] *
                ( - 2.0f*y*z/(y*y+z*z)/(y*y+z*z) );
            }
          }
        }
        else{
          float r2 = (j*cellSize_Y)*(j*cellSize_Y) + (k*cellSize_Z)*(k*cellSize_Z);
          result += cellSize_Y * cellSize_Z * 
                    (- 2.0f* (j*cellSize_Y) * (k*cellSize_Y)/r2/r2);
        }

//         if (r2_int<400){
//           for (int cntb=0; cntb<10; cntb++)
//           for (int cntc=0; cntc<10; cntc++){
//             for (int cnt2=0; cnt2<10; cnt2++){
//               float y = j * cellSize_Y + dev_qd_P_10_Y[cnt2] + dev_qd_P_10_Y[cntb];
//               for (int cnt3=0; cnt3<10; cnt3++){
//                 float z = k * cellSize_Z + dev_qd_P_10_Z[cnt3] + dev_qd_P_10_Z[cntc];
//                 result += 1.0f/4.0f * cellSize_Y * cellSize_Z / 4.0f * dev_qd_W_10[cnt2] * dev_qd_W_10[cnt3] *
//                   ( - 2.0f*y*z/(y*y+z*z)/(y*y+z*z) );
//               }
//             }
//           }
//         }
//         else{
//           float r2 = (j*cellSize_Y)*(j*cellSize_Y) + (k*cellSize_Z)*(k*cellSize_Z);
//           result += cellSize_Y * cellSize_Z * 
//                     (- 2.0f* (j*cellSize_Y) * (k*cellSize_Y)/r2/r2);
//         }

      }
   }
  // ______________________________________________________________________________________________


  // for elements in Kernel component gzz _________________________________________________________
    if (co1==2 && co2==2){
      for(int cntb=-repetition_Y; cntb<=repetition_Y; cntb++)
      for(int cntc=-repetition_Z; cntc<=repetition_Z; cntc++){

        int j = b + cntb*Nkernel_Y/2;
        int k = c + cntc*Nkernel_Z/2;
        int r2_int = j*j+k*k;

        if (r2_int<400){
          for (int cnt2=0; cnt2<10; cnt2++){
            float y = j * cellSize_Y + dev_qd_P_10_Y[cnt2];
            for (int cnt3=0; cnt3<10; cnt3++){
              float z = k * cellSize_Z + dev_qd_P_10_Z[cnt3];
              result += cellSize_Y * cellSize_Z / 4.0f * dev_qd_W_10[cnt2] * dev_qd_W_10[cnt3] *
                ( 1.0f/(y*y+z*z) - 2.0f*z*z/(y*y+z*z)/(y*y+z*z) );
            }
          }
        }
        else{
          float r2 = (j*cellSize_Y)*(j*cellSize_Y) + (k*cellSize_Z)*(k*cellSize_Z);
          result += cellSize_Y * cellSize_Z * 
                    (1.0f/ r2 - 2.0f* (k*cellSize_Y) * (k*cellSize_Y)/r2/r2);
        }
/*        if (r2_int<400){
          for (int cntb=0; cntb<10; cntb++)
          for (int cntc=0; cntc<10; cntc++){
            for (int cnt2=0; cnt2<10; cnt2++){
              float y = j * cellSize_Y + dev_qd_P_10_Y[cnt2] + dev_qd_P_10_Y[cntb];
              for (int cnt3=0; cnt3<10; cnt3++){
                float z = k * cellSize_Z + dev_qd_P_10_Z[cnt3] + dev_qd_P_10_Z[cntc];
                result += 1.0f/4.0f * cellSize_Y * cellSize_Z / 4.0f * dev_qd_W_10[cnt2] * dev_qd_W_10[cnt3] *
                  ( 1.0f/(y*y+z*z) - 2.0f*z*z/(y*y+z*z)/(y*y+z*z) );
              }
            }
          }
        }
        else{
          float r2 = (j*cellSize_Y)*(j*cellSize_Y) + (k*cellSize_Z)*(k*cellSize_Z);
          result += cellSize_Y * cellSize_Z * 
                    (1.0f/ r2 - 2.0f* (k*cellSize_Y) * (k*cellSize_Y)/r2/r2);
        }*/

      }
    }
  // ______________________________________________________________________________________________
  
  result *= -1.0f/2.0f/3.14159265f;
  return( result );
}

__global__ void _gpu_init_kernel_elements_micromag2d(float *data, int Nkernel_Y, int Nkernel_Z, int co1, int co2, float cellSize_Y, float cellSize_Z, int repetition_Y, int repetition_Z, float *dev_qd_P_10, float *dev_qd_W_10){
  
  int j = blockIdx.x;
  int k = blockIdx.y; 

  int N2 = Nkernel_Z;

    data[            j*N2 +           k] = _gpu_get_kernel_element_micromag2d(Nkernel_Y, Nkernel_Z, co1, co2,  j,  k, cellSize_Y, cellSize_Z, repetition_Y, repetition_Z, dev_qd_P_10, dev_qd_W_10);
  if (j>0)
    data[(Nkernel_Y-j)*N2 +           k] = _gpu_get_kernel_element_micromag2d(Nkernel_Y, Nkernel_Z, co1, co2, -j,  k, cellSize_Y, cellSize_Z, repetition_Y, repetition_Z, dev_qd_P_10, dev_qd_W_10);
  if (k>0) 
    data[            j*N2 + Nkernel_Z-k] = _gpu_get_kernel_element_micromag2d(Nkernel_Y, Nkernel_Z, co1, co2,  j, -k, cellSize_Y, cellSize_Z, repetition_Y, repetition_Z, dev_qd_P_10, dev_qd_W_10);
  if (j>0 && k>0) 
    data[(Nkernel_Y-j)*N2 + Nkernel_Z-k] = _gpu_get_kernel_element_micromag2d(Nkernel_Y, Nkernel_Z, co1, co2, -j, -k, cellSize_Y, cellSize_Z, repetition_Y, repetition_Z, dev_qd_P_10, dev_qd_W_10);

  return;
}

void gpu_init_kernel_elements_micromag2d(int co1, int co2, int *kernelSize, float *cellSize, int *repetition){

  // initialization Gauss quadrature points for integrations ______________________________________
    float *dev_qd_W_10 = new_gpu_array(10);
    float *dev_qd_P_10 = new_gpu_array(2*10);
    initialize_Gauss_quadrature_on_gpu_micromag2d(dev_qd_W_10, dev_qd_P_10, cellSize);
  // ______________________________________________________________________________________________
  
  int kernelN = kernelSize[Y]*kernelSize[Z];                              // size of a kernel component without zeros
  float *data = new_gpu_array(kernelN);                                   // to store kernel component without zeros (input of fft routine)
 
  dim3 gridsize(kernelSize[Y]/2,kernelSize[Z]/2, 1);
  dim3 blocksize(1,1,1);
  check3dconf(gridsize, blocksize);
    
  _gpu_init_kernel_elements_micromag2d<<<gridsize, blocksize>>>(data, kernelSize[Y], kernelSize[Z], co1, co2, cellSize[Y], cellSize[Z], repetition[Y], repetition[Z], dev_qd_P_10, dev_qd_W_10);
  gpu_sync();

  cudaFree (dev_qd_W_10);
  cudaFree (dev_qd_P_10);

  write_tensor_pieces(3, kernelSize, data, stdout);
  cudaFree (data);

  return;
}




void initialize_Gauss_quadrature_on_gpu_micromag2d(float *dev_qd_W_10, float *dev_qd_P_10, float *cellSize){

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
    float *host_qd_P_10 =  (float *) calloc (2*10, sizeof(float));
    get_Quad_Points_micromag2d(&host_qd_P_10[ 0], std_qd_P_10, 10, -0.5f*cellSize[Y], 0.5f*cellSize[Y]);
    get_Quad_Points_micromag2d(&host_qd_P_10[10], std_qd_P_10, 10, -0.5f*cellSize[Z], 0.5f*cellSize[Z]);
  // ______________________________________________________________________________________________

  // copy to the quadrature points and weights to the device ______________________________________
    memcpy_to_gpu (host_qd_W_10, dev_qd_W_10, 10);
    memcpy_to_gpu (host_qd_P_10, dev_qd_P_10, 2*10);
  // ______________________________________________________________________________________________

  free (std_qd_P_10);
  free (host_qd_P_10);
  free (host_qd_W_10);

  return;
}

void get_Quad_Points_micromag2d(float *gaussQP, float *stdGaussQP, int qOrder, double a, double b){

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
// //  _gpu_init_Greens_kernel_elements<<<gridsize1, blocksize1>>>(dev_temp, Nkernel[X], Nkernel[Y], Nkernel[Z], testco1, testco2, cellSize[X], cellSize_Y, cellSize_Z, cst, repetition[X], repetition[Y], repetition[Z], dev_qd_P_10, dev_qd_W_10);
//   _gpu_init_Greens_kernel_elements<<<gridsize1, blocksize1>>>(dev_temp, kernelSize[X], kernelSize[Y], kernelSize[Z], kernelStorageSize[Z], testco1, testco2, cellSize[X], cellSize_Y, cellSize_Z, repetition[X], repetition[Y], repetition[Z], dev_qd_P_10, dev_qd_W_10);
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
//   _gpu_extract_real_parts<<<gridsize2, blocksize2>>>(&dev_kernel->list[testrang*kernelStorageN/2], dev_temp, 0, kernelStorageN/2);
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
