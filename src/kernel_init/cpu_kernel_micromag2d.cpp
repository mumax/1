#include "tensor.h"
#include <stdio.h>
#include "cpu_kernel_micromag2d.h"
#include "cpukern.h"

#ifdef __cplusplus
extern "C" {
#endif


void cpu_init_kernel_elements_micromag2d(int co1, int co2, int *kernelSize, float *cellSize, int *repetition){

  // initialization Gauss quadrature points for integrations ______________________________________
    float *qd_W_10 = (float*)calloc(10, sizeof(float));
    float *qd_P_10 = (float*)calloc(2*10, sizeof(float));
    initialize_Gauss_quadrature_micromag2d(qd_W_10, qd_P_10, cellSize);
  // ______________________________________________________________________________________________
  
  int kernelN = kernelSize[Y]*kernelSize[Z];       // size of a kernel component without zeros
  float *data = new_cpu_array(kernelN);       // to store kernel component without zeros (input of fft routine)
 
  _cpu_init_kernel_elements_micromag2d(data, kernelSize, co1, co2, cellSize, repetition, qd_P_10, qd_W_10);

  free (qd_P_10);
  free (qd_W_10);
  
  // TODO
  //write_tensor_pieces(3, kernelSize, data, stdout);
  print_tensor(as_tensorN(data, 3, kernelSize));
  free (data);
  
  return;
}


typedef struct{
  float *cellSize, *data, *qd_P_10, *qd_W_10;
  int *Nkernel, *repetition; 
  int co1, co2;
} _cpu_init_kernel_elements_micromag2d_arg;


float _cpu_get_kernel_element_micromag2d(_cpu_init_kernel_elements_micromag2d_arg *arg, int b, int c){

  int *Nkernel = arg->Nkernel;
  int co1 = arg->co1;
  int co2 = arg->co2;
  float *cellSize = arg->cellSize;
  int *repetition = arg->repetition;
  float *qd_P_10 = arg->qd_P_10;
  float *qd_W_10 = arg->qd_W_10;
  
  float result = 0.0f;
  float *qd_P_10_Y = &qd_P_10[ 0];
  float *qd_P_10_Z = &qd_P_10[10];

  // for elements in Kernel component gyy _________________________________________________________
    if (co1==1 && co2==1){
      for(int cntb=-repetition[Y]; cntb<=repetition[Y]; cntb++)
      for(int cntc=-repetition[Z]; cntc<=repetition[Z]; cntc++){

        int j = b + cntb*Nkernel[Y]/2;
        int k = c + cntc*Nkernel[Z]/2;
        int r2_int = j*j+k*k;

        if (r2_int<400){
          for (int cnt2=0; cnt2<10; cnt2++){
            float y = j * cellSize[Y] + qd_P_10_Y[cnt2];
            for (int cnt3=0; cnt3<10; cnt3++){
              float z = k * cellSize[Z] + qd_P_10_Z[cnt3];
              result += cellSize[Y] * cellSize[Z] / 4.0f * qd_W_10[cnt2] * qd_W_10[cnt3] *
                ( 1.0f/(y*y+z*z) - 2.0f*y*y/(y*y+z*z)/(y*y+z*z) );
            }
          }
        }
        else{
          float r2 = (j*cellSize[Y])*(j*cellSize[Y]) + (k*cellSize[Z])*(k*cellSize[Z]);
          result += cellSize[Y] * cellSize[Z] * 
                    (1.0f/ r2 - 2.0f* (j*cellSize[Y]) * (j*cellSize[Y])/r2/r2 );
        }
        
/*        if (r2_int<400){
          for (int cntb=0; cntb<10; cntb++)
          for (int cntc=0; cntc<10; cntc++){
            for (int cnt2=0; cnt2<10; cnt2++){
              float y = j * cellSize[Y] + qd_P_10_Y[cnt2] + qd_P_10_Y[cntb];
              for (int cnt3=0; cnt3<10; cnt3++){
                float z = k * cellSize[Z] + qd_P_10_Z[cnt3] + qd_P_10_Z[cntc];
                result += 1.0f/4.0f * cellSize[Y] * cellSize[Z] / 4.0f * qd_W_10[cnt2] * qd_W_10[cnt3] *
                  ( 1.0f/(y*y+z*z) - 2.0f*y*y/(y*y+z*z)/(y*y+z*z) );
              }
            }
          }
        }
        else{
          float r2 = (j*cellSize[Y])*(j*cellSize[Y]) + (k*cellSize[Z])*(k*cellSize[Z]);
          result += cellSize[Y] * cellSize[Z] * 
                    (1.0f/ r2 - 2.0f* (j*cellSize[Y]) * (j*cellSize[Y])/r2/r2 );
        }*/
        
      }
    }
  // ______________________________________________________________________________________________


  // for elements in Kernel component gyz _________________________________________________________
    if (co1==1 && co2==2){
      for(int cntb=-repetition[Y]; cntb<=repetition[Y]; cntb++)
      for(int cntc=-repetition[Z]; cntc<=repetition[Z]; cntc++){

        int j = b + cntb*Nkernel[Y]/2;
        int k = c + cntc*Nkernel[Z]/2;
        int r2_int = j*j+k*k;

        if (r2_int<400){
          for (int cnt2=0; cnt2<10; cnt2++){
            float y = j * cellSize[Y] + qd_P_10_Y[cnt2];
            for (int cnt3=0; cnt3<10; cnt3++){
              float z = k * cellSize[Z] + qd_P_10_Z[cnt3];
              result += cellSize[Y] * cellSize[Z] / 4.0f * qd_W_10[cnt2] * qd_W_10[cnt3] *
                ( - 2.0f*y*z/(y*y+z*z)/(y*y+z*z) );
            }
          }
        }
        else{
          float r2 = (j*cellSize[Y])*(j*cellSize[Y]) + (k*cellSize[Z])*(k*cellSize[Z]);
          result += cellSize[Y] * cellSize[Z] * 
                    (- 2.0f* (j*cellSize[Y]) * (k*cellSize[Y])/r2/r2);
        }
        
//         if (r2_int<400){
//           for (int cntb=0; cntb<10; cntb++)
//           for (int cntc=0; cntc<10; cntc++){
//             for (int cnt2=0; cnt2<10; cnt2++){
//               float y = j * cellSize[Y] + qd_P_10_Y[cnt2] + qd_P_10_Y[cntb];
//               for (int cnt3=0; cnt3<10; cnt3++){
//                 float z = k * cellSize[Z] + qd_P_10_Z[cnt3] + qd_P_10_Z[cntc];
//                 result += 1.0f/4.0f * cellSize[Y] * cellSize[Z] / 4.0f * qd_W_10[cnt2] * qd_W_10[cnt3] *
//                   ( - 2.0f*y*z/(y*y+z*z)/(y*y+z*z) );
//               }
//             }
//           }
//         }
//         else{
//           float r2 = (j*cellSize[Y])*(j*cellSize[Y]) + (k*cellSize[Z])*(k*cellSize[Z]);
//           result += cellSize[Y] * cellSize[Z] * 
//                     (- 2.0f* (j*cellSize[Y]) * (k*cellSize[Y])/r2/r2);
//         }

      
      }
   }
  // ______________________________________________________________________________________________


  // for elements in Kernel component gzz _________________________________________________________
    if (co1==2 && co2==2){
      for(int cntb=-repetition[Y]; cntb<=repetition[Y]; cntb++)
      for(int cntc=-repetition[Z]; cntc<=repetition[Z]; cntc++){

        int j = b + cntb*Nkernel[Y]/2;
        int k = c + cntc*Nkernel[Z]/2;
        int r2_int = j*j+k*k;

        if (r2_int<400){
          for (int cnt2=0; cnt2<10; cnt2++){
            float y = j * cellSize[Y] + qd_P_10_Y[cnt2];
            for (int cnt3=0; cnt3<10; cnt3++){
              float z = k * cellSize[Z] + qd_P_10_Z[cnt3];
              result += cellSize[Y] * cellSize[Z] / 4.0f * qd_W_10[cnt2] * qd_W_10[cnt3] *
                ( 1.0f/(y*y+z*z) - 2.0f*z*z/(y*y+z*z)/(y*y+z*z) );
            }
          }
        }
        else{
          float r2 = (j*cellSize[Y])*(j*cellSize[Y]) + (k*cellSize[Z])*(k*cellSize[Z]);
          result += cellSize[Y] * cellSize[Z] * 
                    (1.0f/ r2 - 2.0f* (k*cellSize[Y]) * (k*cellSize[Y])/r2/r2);
        }

/*        if (r2_int<400){
          for (int cntb=0; cntb<10; cntb++)
          for (int cntc=0; cntc<10; cntc++){
            for (int cnt2=0; cnt2<10; cnt2++){
              float y = j * cellSize[Y] + qd_P_10_Y[cnt2] + qd_P_10_Y[cntb];
              for (int cnt3=0; cnt3<10; cnt3++){
                float z = k * cellSize[Z] + qd_P_10_Z[cnt3] + qd_P_10_Z[cntc];
                result += 1.0f/4.0f * cellSize[Y] * cellSize[Z] / 4.0f * qd_W_10[cnt2] * qd_W_10[cnt3] *
                  ( 1.0f/(y*y+z*z) - 2.0f*z*z/(y*y+z*z)/(y*y+z*z) );
              }
            }
          }
        }
        else{
          float r2 = (j*cellSize[Y])*(j*cellSize[Y]) + (k*cellSize[Z])*(k*cellSize[Z]);
          result += cellSize[Y] * cellSize[Z] * 
                    (1.0f/ r2 - 2.0f* (k*cellSize[Y]) * (k*cellSize[Y])/r2/r2);
        }*/
      
      }
    }
  // ______________________________________________________________________________________________
  
  result *= -1.0f/2.0f/3.14159265f;
  return( result );
}

void _cpu_init_kernel_elements_micromag2d_t(int id){
  
  _cpu_init_kernel_elements_micromag2d_arg *arg = (_cpu_init_kernel_elements_micromag2d_arg *) func_arg;

  int start, stop;
  init_start_stop (&start, &stop, id, arg->Nkernel[Y]/2);

  int N2 = arg->Nkernel[Z];
  for (int j=start; j<stop; j++)
    for (int k=0; k<N2/2; k++){
        arg->data[                  j*N2 +                 k] = _cpu_get_kernel_element_micromag2d(arg, j, k);
      if (j>0)
        arg->data[(arg->Nkernel[Y]-j)*N2 +                 k] = _cpu_get_kernel_element_micromag2d(arg, -j, k);
      if (k>0) 
        arg->data[                  j*N2 + arg->Nkernel[Z]-k] = _cpu_get_kernel_element_micromag2d(arg, j, -k);
      if (j>0 && k>0) 
        arg->data[(arg->Nkernel[Y]-j)*N2 + arg->Nkernel[Z]-k] = _cpu_get_kernel_element_micromag2d(arg, -j, -k);
    }
    
  return;
}

void _cpu_init_kernel_elements_micromag2d(float *data, int *Nkernel, int co1, int co2, float *cellSize, int *repetition, float *qd_P_10, float *qd_W_10){

  _cpu_init_kernel_elements_micromag2d_arg args;

  args.data = data;
  args.Nkernel = Nkernel;
  args.co1 = co1;
  args.co2 = co2;
  args.cellSize = cellSize;
  args.repetition = repetition;
  args.qd_P_10 = qd_P_10;
  args.qd_W_10 = qd_W_10;

  func_arg = (void *) (&args);

  thread_Wrapper(_cpu_init_kernel_elements_micromag2d_t);

  return;
}



void initialize_Gauss_quadrature_micromag2d(float *std_qd_W_10, float *mapped_qd_P_10, float *cellSize){

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
    get_Quad_Points_micromag2d(&mapped_qd_P_10[0], std_qd_P_10, 10, -0.5f*cellSize[Y], 0.5f*cellSize[Y]);
    get_Quad_Points_micromag2d(&mapped_qd_P_10[10], std_qd_P_10, 10, -0.5f*cellSize[Z], 0.5f*cellSize[Z]);
  // ______________________________________________________________________________________________


  free (std_qd_P_10);

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
//   gpu_zero(temp, kernelStorageN);
//   gpu_sync();
// //  _gpu_init_Greens_kernel_elements<<<gridsize1, blocksize1>>>(temp, Nkernel[X], Nkernel[Y], Nkernel[Z], testco1, testco2, cellSize[X], cellSize[Y], cellSize[Z], cst, repetition[X], repetition[Y], repetition[Z], qd_P_10, qd_W_10);
//   _gpu_init_Greens_kernel_elements<<<gridsize1, blocksize1>>>(temp, kernelSize[X], kernelSize[Y], kernelSize[Z], kernelStorageSize[Z], testco1, testco2, cellSize[X], cellSize[Y], cellSize[Z], repetition[X], repetition[Y], repetition[Z], qd_P_10, qd_W_10);
//   gpu_sync();
// 
//   memcpy_from_gpu(temp, host_temp, kernelStorageN);
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
//   memcpy_from_gpu(temp, host_temp, kernelStorageN);
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
//   _gpu_extract_real_parts<<<gridsize2, blocksize2>>>(&kernel->list[testrang*kernelStorageN/2], temp, 0, kernelStorageN/2);
//   gpu_sync();
//   fprintf(stderr, "\nkernel elements (transformed, real parts), co: %d, %d:\n", testco1, testco2);
//   memcpy_from_gpu(&kernel->list[testrang*kernelStorageN/2], host_temp2, kernelStorageN/2);
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
