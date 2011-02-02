#include "tensor.h"
#include "gputil.h"
//#include "gpufft2.h"
#include "gpu_fftbig.h"
#include "gpu_fft6.h"
#include "assert.h"
#include "timer.h"
#include <stdio.h>
#include "gpu_conf.h"
#include "gpu_micromag2d_kernel.h"

#ifdef __cplusplus
extern "C" {
#endif



// tensor *gpu_micromag2d_kernel(param* p){
void gpu_kernel_micromag2d(int *kernelSize, float *cellsize, int exchType, int *exchInConv, int *repetition){
  
  // check input + allocate tensor on device ______________________________________________________
    int kernelStorageN = p->kernelSize[Y] * (pkernelSize[Z]+2);
    tensor *dev_kernel;
    dev_kernel = as_tensor(new_gpu_array(3*kernelStorageN/2), 2, 3, kernelStorageN/2);  // only real parts!!
  // ______________________________________________________________________________________________


  // initialization Gauss quadrature points for integrations + copy to gpu ________________________
    float *dev_qd_W_10 = new_gpu_array(10);
    float *dev_qd_P_10 = new_gpu_array(2*10);
    initialize_Gauss_quadrature_on_gpu_micromag2d(dev_qd_W_10, dev_qd_P_10, cellSize);
  // ______________________________________________________________________________________________
  
  
  // Plan initialization of FFTs and initialization of the kernel _________________________________
    gpuFFT3dPlan* kernel_plan = new_gpuFFT3dPlan_padded(kernelSize, kernelSize);   //works also for 2d transforms!
    gpu_init_and_FFT_Greens_kernel_elements_micromag2d(dev_kernel->list, kernelSize, exchType, exchInConv, cellSize, repetition, dev_qd_P_10, dev_qd_W_10, kernel_plan);
  // ______________________________________________________________________________________________ 

  cudaFree (dev_qd_W_10);
  cudaFree (dev_qd_P_10);

  write_tensor(tensor* t, FILE* out);
  return;
//  return (dev_kernel);   ///> moet de kernel hier gefreed worden?
}

/// @todo argument defining which Greens function should be added
/// remark: number of FD cells in a dimension can not be odd if no zero padding!!
void gpu_init_and_FFT_Greens_kernel_elements_micromag2d(float *dev_kernel, int *kernelSize, int exchType, int *exchInConv, float *FD_cell_size, int *repetition, float *dev_qd_P_10, float *dev_qd_W_10, gpuFFT3dPlan* kernel_plan){

  
  int kernelN = kernelSize[Y]*kernelSize[Z];                              // size of a kernel component without zeros
  float *dev_temp1 = new_gpu_array(kernelN);                              // temp array on device for storage of kernel component without zeros (input of fft routine)
  int kernelStorageN = kernelSize[Y]*(kernelSize[Z]+2);                   // size of a zero padded kernel component
  float *dev_temp2 = new_gpu_array(kernelStorageN);                       // temp array on device for storage of zero padded kernel component (output of fft routine)
 
  // Define gpugrids and blocks ___________________________________________________________________
    dim3 gridsize1(kernelSize[Y]/2,kernelSize[Z]/2, 1);
    dim3 blocksize1(1,1,1);
    check3dconf(gridsize1, blocksize1);
    
    int N = kernelStorageN/2;
    dim3 gridsize2, blocksize2;
    make1dconf(N, &gridsize2, &blocksize2);
  // ______________________________________________________________________________________________
  
  // Main function operations _____________________________________________________________________
    int rank0 = 0;                                      // defines the first rank of the Greens kernel: [yy, yz, zz]
    for (int co1=1; co1<3; co1++){                      // for a Greens kernel component [co1,co2]:
      for (int co2=co1; co2<3; co2++){
          // Put all elements in 'dev_temp1' to zero.
        gpu_zero(dev_temp1, kernelN);    
        gpu_sync();
        // Fill in the elements.
        _gpu_init_Greens_kernel_elements_micromag2d<<<gridsize1, blocksize1>>>(dev_temp1, kernelSize[Y], kernelSize[Z], exchType, exchInConv[Y], exchInConv[Z], co1, co2, FD_cell_size[Y], FD_cell_size[Z], repetition[Y], repetition[Z], dev_qd_P_10, dev_qd_W_10);
        gpu_sync();
        // Fourier transform the kernel component.
        gpuFFT3dPlan_forward(kernel_plan, dev_temp1, dev_temp2); 
        gpu_sync();
        // Copy the real parts to the corresponding place in the dev_kernel tensor.
        _gpu_extract_real_parts_micromag2d<<<gridsize2, blocksize2>>>(&dev_kernel[rank0*kernelStorageN/2], dev_temp2, N);
        gpu_sync();
        rank0++;                                        // get ready for next component
      }
    } 
  // ______________________________________________________________________________________________

  cudaFree (dev_temp1);
  cudaFree (dev_temp2);
  
  return;
}



__global__ void _gpu_init_Greens_kernel_elements_micromag2d(float *dev_temp, int Nkernel_Y, int Nkernel_Z, int exchType, int exchInConv_Y, int exchInConv_Z, int co1, int co2, float FD_cell_size_Y, float FD_cell_size_Z, int repetition_Y, int repetition_Z, float *dev_qd_P_10, float *dev_qd_W_10){
  
  int j = blockIdx.x;
  int k = blockIdx.y; 

  int N2 = Nkernel_Z;

    dev_temp[            j*N2 +           k] = _gpu_get_Greens_element_micromag2d(Nkernel_Y, Nkernel_Z, exchType, exchInConv_Y, exchInConv_Z, co1, co2,  j,  k, FD_cell_size_Y, FD_cell_size_Z, repetition_Y, repetition_Z, dev_qd_P_10, dev_qd_W_10);
  if (j>0)
    dev_temp[(Nkernel_Y-j)*N2 +           k] = _gpu_get_Greens_element_micromag2d(Nkernel_Y, Nkernel_Z, exchType, exchInConv_Y, exchInConv_Z, co1, co2, -j,  k, FD_cell_size_Y, FD_cell_size_Z, repetition_Y, repetition_Z, dev_qd_P_10, dev_qd_W_10);
  if (k>0) 
    dev_temp[            j*N2 + Nkernel_Z-k] = _gpu_get_Greens_element_micromag2d(Nkernel_Y, Nkernel_Z, exchType, exchInConv_Y, exchInConv_Z, co1, co2,  j, -k, FD_cell_size_Y, FD_cell_size_Z, repetition_Y, repetition_Z, dev_qd_P_10, dev_qd_W_10);
  if (j>0 && k>0) 
    dev_temp[(Nkernel_Y-j)*N2 + Nkernel_Z-k] = _gpu_get_Greens_element_micromag2d(Nkernel_Y, Nkernel_Z, exchType, exchInConv_Y, exchInConv_Z, co1, co2, -j, -k, FD_cell_size_Y, FD_cell_size_Z, repetition_Y, repetition_Z, dev_qd_P_10, dev_qd_W_10);

  return;
}



__device__ float _gpu_get_Greens_element_micromag2d(int Nkernel_Y, int Nkernel_Z, int exchType, int exchInConv_Y, int exchInConv_Z, int co1, int co2, int b, int c, float FD_cell_size_Y, float FD_cell_size_Z, int repetition_Y, int repetition_Z, float *dev_qd_P_10, float *dev_qd_W_10){

  float result = 0.0f;
  float *dev_qd_P_10_Y = &dev_qd_P_10[ 0];
  float *dev_qd_P_10_Z = &dev_qd_P_10[10];
  float dim_inverse = 1.0f/( (float) Nkernel_Y*Nkernel_Z  );
  

  // for elements in Kernel component gyy _________________________________________________________
    if (co1==1 && co2==1){
      for(int cntb=-repetition_Y; cntb<=repetition_Y; cntb++)
      for(int cntc=-repetition_Z; cntc<=repetition_Z; cntc++){

        int j = b + cntb*Nkernel_Y/2;
        int k = c + cntc*Nkernel_Z/2;
        int r2_int = j*j+k*k;

        if (r2_int<400){
          for (int cnt2=0; cnt2<10; cnt2++){
            float y = j * FD_cell_size_Y + dev_qd_P_10_Y[cnt2];
            for (int cnt3=0; cnt3<10; cnt3++){
              float z = k * FD_cell_size_Z + dev_qd_P_10_Z[cnt3];
              result += FD_cell_size_Y * FD_cell_size_Z / 4.0f * dev_qd_W_10[cnt2] * dev_qd_W_10[cnt3] *
                ( 1.0f/(y*y+z*z) - 2.0f*y*y/(y*y+z*z)/(y*y+z*z) );
            }
          }
        }
        else{
          float r2 = (j*FD_cell_size_Y)*(j*FD_cell_size_Y) + (k*FD_cell_size_Z)*(k*FD_cell_size_Z);
          result += FD_cell_size_Y * FD_cell_size_Z * 
                    (1.0f/ r2 - 2.0f* (j*FD_cell_size_Y) * (j*FD_cell_size_Y)/r2/r2 );
        }
      }
      result *= -1.0f/2.0f/3.14159265f*dim_inverse;

      if (exchType == EXCH_6NGBR && exchInConv_Y==1){
        if (b== 0 && c== 0)  result -= 2.0f/FD_cell_size_Y/FD_cell_size_Y + 2.0f/FD_cell_size_Z/FD_cell_size_Z;
        if (b== 1 && c== 0)  result += 2.0f/FD_cell_size_Y/FD_cell_size_Y;
        if (b==-1 && c== 0)  result += 2.0f/FD_cell_size_Y/FD_cell_size_Y;
        if (b== 0 && c== 1)  result += 2.0f/FD_cell_size_Z/FD_cell_size_Z;
        if (b== 0 && c==-1)  result += 2.0f/FD_cell_size_Z/FD_cell_size_Z;
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
            float y = j * FD_cell_size_Y + dev_qd_P_10_Y[cnt2];
            for (int cnt3=0; cnt3<10; cnt3++){
              float z = k * FD_cell_size_Z + dev_qd_P_10_Z[cnt3];
              result += FD_cell_size_Y * FD_cell_size_Z / 4.0f * dev_qd_W_10[cnt2] * dev_qd_W_10[cnt3] *
                ( - 2.0f*y*z/(y*y+z*z)/(y*y+z*z) );
            }
          }
        }
        else{
          float r2 = (j*FD_cell_size_Y)*(j*FD_cell_size_Y) + (k*FD_cell_size_Z)*(k*FD_cell_size_Z);
          result += FD_cell_size_Y * FD_cell_size_Z * 
                    (- 2.0f* (j*FD_cell_size_Y) * (k*FD_cell_size_Y)/r2/r2);
        }
      }
      result *= -1.0f/2.0f/3.14159265f*dim_inverse;

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
            float y = j * FD_cell_size_Y + dev_qd_P_10_Y[cnt2];
            for (int cnt3=0; cnt3<10; cnt3++){
              float z = k * FD_cell_size_Z + dev_qd_P_10_Z[cnt3];
              result += FD_cell_size_Y * FD_cell_size_Z / 4.0f * dev_qd_W_10[cnt2] * dev_qd_W_10[cnt3] *
                ( 1.0f/(y*y+z*z) - 2.0f*z*z/(y*y+z*z)/(y*y+z*z) );
            }
          }
        }
        else{
          float r2 = (j*FD_cell_size_Y)*(j*FD_cell_size_Y) + (k*FD_cell_size_Z)*(k*FD_cell_size_Z);
          result += FD_cell_size_Y * FD_cell_size_Z * 
                    (1.0f/ r2 - 2.0f* (k*FD_cell_size_Y) * (k*FD_cell_size_Y)/r2/r2);
        }
      }
      result *= -1.0f/2.0f/3.14159265f*dim_inverse;

      if (exchType == EXCH_6NGBR && exchInConv_Z==1){
        if (b== 0 && c== 0)  result -= 2.0f/FD_cell_size_Y/FD_cell_size_Y + 2.0f/FD_cell_size_Z/FD_cell_size_Z;
        if (b== 1 && c== 0)  result += 2.0f/FD_cell_size_Y/FD_cell_size_Y;
        if (b==-1 && c== 0)  result += 2.0f/FD_cell_size_Y/FD_cell_size_Y;
        if (b== 0 && c== 1)  result += 2.0f/FD_cell_size_Z/FD_cell_size_Z;
        if (b== 0 && c==-1)  result += 2.0f/FD_cell_size_Z/FD_cell_size_Z;
      }
    }
  // ______________________________________________________________________________________________
  
  return( result );       //correct for scaling factor in FFTs
}



__global__ void _gpu_extract_real_parts_micromag2d(float *dev_kernel_array, float *dev_temp, int N){

  int e = threadindex;

  if (e<N)
    dev_kernel_array[e] = dev_temp[2*e];

  return;
}



void initialize_Gauss_quadrature_on_gpu_micromag2d(float *dev_qd_W_10, float *dev_qd_P_10, float *FD_cell_size){

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
    get_Quad_Points_micromag2d(&host_qd_P_10[ 0], std_qd_P_10, 10, -0.5f*FD_cell_size[Y], 0.5f*FD_cell_size[Y]);
    get_Quad_Points_micromag2d(&host_qd_P_10[10], std_qd_P_10, 10, -0.5f*FD_cell_size[Z], 0.5f*FD_cell_size[Z]);
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
// //  _gpu_init_Greens_kernel_elements<<<gridsize1, blocksize1>>>(dev_temp, Nkernel[X], Nkernel[Y], Nkernel[Z], testco1, testco2, FD_cell_size[X], FD_cell_size[Y], FD_cell_size[Z], cst, repetition[X], repetition[Y], repetition[Z], dev_qd_P_10, dev_qd_W_10);
//   _gpu_init_Greens_kernel_elements<<<gridsize1, blocksize1>>>(dev_temp, kernelSize[X], kernelSize[Y], kernelSize[Z], kernelStorageSize[Z], testco1, testco2, FD_cell_size[X], FD_cell_size[Y], FD_cell_size[Z], repetition[X], repetition[Y], repetition[Z], dev_qd_P_10, dev_qd_W_10);
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
