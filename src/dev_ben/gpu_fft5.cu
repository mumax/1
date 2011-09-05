#include "gputil.h"
#include <cufft.h>
#include "gpu_transpose.h"
#include "gpu_transpose2.h"
#include "gpu_safe.h"
#include "gpu_fft4.h"
#include "gpu_fft5.h"
#include "gpu_fftbig.h"
#include "gpu_conf.h"
#include "timer.h"

#ifdef __cplusplus
extern "C" {
#endif

gpuFFT3dPlan_big* new_gpuFFT3dPlan_padded_big(int* size, int* paddedSize){

  int N0 = size[X];
  int N1 = size[Y];
  int N2 = size[Z];
  
  assert(paddedSize[X] > 0);
  assert(paddedSize[Y] > 1);
  assert(paddedSize[Z] > 1);
  
  gpuFFT3dPlan_big* plan = (gpuFFT3dPlan_big*)malloc(sizeof(gpuFFT3dPlan_big));
  
  plan->size = (int*)calloc(3, sizeof(int));    ///@todo not int* but int[3]
  plan->paddedSize = (int*)calloc(3, sizeof(int));
  plan->paddedStorageSize = (int*)calloc(3, sizeof(int));
  
//   int* paddedSize = plan->paddedSize;
  int* paddedStorageSize = plan->paddedStorageSize;
  
  plan->size[0] = N0; 
  plan->size[1] = N1; 
  plan->size[2] = N2;
  plan->N = N0 * N1 * N2;
  
  plan->paddedSize[X] = paddedSize[X];
  plan->paddedSize[Y] = paddedSize[Y];
  plan->paddedSize[Z] = paddedSize[Z];
  plan->paddedN = plan->paddedSize[0] * plan->paddedSize[1] * plan->paddedSize[2];
  
  plan->paddedStorageSize[X] = plan->paddedSize[X];
  plan->paddedStorageSize[Y] = plan->paddedSize[Y];
  plan->paddedStorageSize[Z] = plan->paddedSize[Z] + 2;
  plan->paddedStorageN = paddedStorageSize[X] * paddedStorageSize[Y] * paddedStorageSize[Z];

  plan->fwPlanZ = (bigfft *) malloc(sizeof(bigfft));
  plan->invPlanZ = (bigfft *) malloc(sizeof(bigfft));
  plan->planY = (bigfft *) malloc(sizeof(bigfft));
  plan->planX = (bigfft *) malloc(sizeof(bigfft));
  
  if ( paddedSize[X]!=size[X] || paddedSize[Y]!=size[Y]){
    init_bigfft(plan->fwPlanZ , paddedSize[Z], paddedSize[Z], plan->paddedStorageSize[Z], CUFFT_R2C, size[X]*size[Y]);
    init_bigfft(plan->invPlanZ, paddedSize[Z], plan->paddedStorageSize[Z], paddedSize[Z], CUFFT_C2R, size[X]*size[Y]);
  }
  else{
    init_bigfft(plan->fwPlanZ , paddedSize[Z], plan->paddedStorageSize[Z], plan->paddedStorageSize[Z], CUFFT_R2C, size[X]*size[Y]);
    init_bigfft(plan->invPlanZ, paddedSize[Z], plan->paddedStorageSize[Z], plan->paddedStorageSize[Z], CUFFT_C2R, size[X]*size[Y]);
  }
  init_bigfft(plan->planY, paddedSize[Y], paddedSize[Y], paddedSize[Y], CUFFT_C2C, paddedStorageSize[Z] * size[X] / 2);
  init_bigfft(plan->planX, paddedSize[X], paddedSize[X], paddedSize[X], CUFFT_C2C, paddedStorageSize[Z] * paddedSize[Y] / 2);
  
  for (int i=0; i<plan->fwPlanZ->Nbatch; i++)
    printf("fwZ: Nbatch: %d, i: %d, batch: %d, batch_index_in: %d, batch_index_out: %d\n", plan->fwPlanZ->Nbatch, i, plan->fwPlanZ->batch[i], plan->fwPlanZ->batch_index_in[i], plan->fwPlanZ->batch_index_out[i]);
  printf("\n");  
  for (int i=0; i<plan->invPlanZ->Nbatch; i++)
    printf("invZ: Nbatch: %d, i: %d, batch: %d, batch_index_in: %d, batch_index_out: %d\n", plan->invPlanZ->Nbatch, i, plan->invPlanZ->batch[i], plan->invPlanZ->batch_index_in[i], plan->invPlanZ->batch_index_out[i]);
  printf("\n");  
  for (int i=0; i<plan->planY->Nbatch; i++)
    printf("Y: Nbatch: %d, i: %d, batch: %d, batch_index_in: %d, batch_index_out: %d\n", plan->planY->Nbatch, i, plan->planY->batch[i], plan->planY->batch_index_in[i], plan->planY->batch_index_out[i]);
  printf("\n");  
  for (int i=0; i<plan->planX->Nbatch; i++)
    printf("X: Nbatch: %d, i: %d, batch: %d, batch_index_in: %d, batch_index_out: %d\n", plan->planX->Nbatch, i, plan->planX->batch[i], plan->planX->batch_index_in[i], plan->planX->batch_index_out[i]);
  printf("\n");  
  
  plan->transp = new_gpu_array(plan->paddedStorageN);
  
  return plan;
}




void gpuFFT3dPlan_forward_big(gpuFFT3dPlan_big* plan, float* input, float* output){
//   timer_start("gpu_plan3d_real_input_forward_exec");

  int* size = plan->size;
  int* pSSize = plan->paddedStorageSize;
  int N0 = pSSize[X];
  int N1 = pSSize[Y];
  int N2 = pSSize[Z]/2; // we treat the complex data as an N0 x N1 x N2 x 2 array
  int N3 = 2;
  
  int half_pSSize = plan->paddedStorageN/2;
  
  //     zero out the output matrix
    gpu_zero(output, plan->paddedStorageN);
  //     padding of the input matrix towards the output matrix
    timer_start("fw_copy_to_pad");
    gpu_copy_to_pad(input, output, size, pSSize);
    timer_stop("fw_copy_to_pad");

  
  float* data = output;
  float* data2 = plan->transp; 

  if ( pSSize[X]!=size[X] || pSSize[Y]!=size[Y]){
      //out of place FFTs in Z-direction from the 0-element towards second half of the zeropadded matrix (out of place: no +2 on input!)
    timer_start("fw_fft_on_z");
    bigfft_execR2C(plan->fwPlanZ, (cufftReal*)data,  (cufftComplex*) (data + half_pSSize));                // it's in data
    gpu_sync();
    timer_stop("fw_fft_on_z");

      // zero out the input data points at the start of the matrix
    gpu_zero(data, size[X]*size[Y]*pSSize[Z]);
    
      // YZ-transpose within the same matrix from the second half of the matrix towards the 0-element
     timer_start("fw_yz_transpose");
    yz_transpose_in_place_fw(data, size, pSSize);                                                          // it's in data
     timer_stop("fw_yz_transpose");

      // in place FFTs in Y-direction
    timer_start("fw_fft_on_y");
    bigfft_execC2C(plan->planY, (cufftComplex*)data,  (cufftComplex*)data, CUFFT_FORWARD);                 // it's in data
    gpu_sync();
    timer_stop("fw_fft_on_y");
  }
  
  else {          //no zero padding in X- and Y direction (e.g. for Greens kernel computations)
      // in place FFTs in Z-direction (there is no zero space to perform them out of place)
    bigfft_execR2C(plan->fwPlanZ, (cufftReal*)data,  (cufftComplex*) (data));                              // it's in data
    gpu_sync();
    
      // YZ-transpose needs to be out of place.
    gpu_transposeYZ_complex(data, data2, N0, N1, N2*N3);                                                   // it's in data2
    
      // perform the FFTs in the Y-direction
    bigfft_execC2C(plan->planY, (cufftComplex*)data2,  (cufftComplex*)data, CUFFT_FORWARD);                // it's in data
    gpu_sync();
  }

  if(N0 > 1){    // not done for 2D transforms
      // XZ transpose still needs to be out of place
    timer_start("fw_xz_transpose");
    gpu_transposeXZ_complex(data, data2, N0, N2, N1*N3);                                                   // it's in data2
    timer_stop("fw_xz_transpose");
 
    
      // out of place FFTs in X-direction
    timer_start("fw_fft_on_x");
    bigfft_execC2C(plan->planX, (cufftComplex*)data2, (cufftComplex*)output, CUFFT_FORWARD);               // it's in output
    gpu_sync();
    timer_stop("fw_fft_on_x");
    
  }

//   timer_stop("gpu_plan3d_real_input_forward_exec");
  
  return;
}




void gpuFFT3dPlan_inverse_big(gpuFFT3dPlan_big* plan, float* input, float* output){
  
//   timer_start("gpu_plan3d_real_input_inverse_exec");
//   printf("start inverse\n");
  int* size = plan->size;
  int* pSSize = plan->paddedStorageSize;
  int N0 = pSSize[X];
  int N1 = pSSize[Y];
  int N2 = pSSize[Z]/2; // we treat the complex data as an N0 x N1 x N2 x 2 array
  int N3 = 2;
  int half_pSSize = plan->paddedStorageN/2;
  
  float* data = input;
  float* data2 = plan->transp; // both the transpose and FFT are out-of-place between data and data2

  if (N0 > 1){
      // out of place FFTs in the X-direction (i.e. no +2 stride on input!)
    timer_start("inv_ff_on_x");
    bigfft_execC2C(plan->planX, (cufftComplex*)data,  (cufftComplex*)data2, CUFFT_INVERSE);                // it's in data2
    gpu_sync();
    timer_stop("inv_ff_on_x");

      // XZ transpose still needs to be out of place
    timer_start("inv_xz_transpose");
    gpu_transposeXZ_complex(data2, data, N1, N2, N0*N3);                                                   // it's in data
    timer_stop("inv_xz_transpose");
  }
  
  if ( pSSize[X]!=size[X] || pSSize[Y]!=size[Y]){
      // in place FFTs in Y-direction
    timer_start("inv_fft_on_y");
    bigfft_execC2C(plan->planY, (cufftComplex*)data,  (cufftComplex*)data, CUFFT_INVERSE);                 // it's in data
    gpu_sync();
    timer_stop("inv_fft_on_y");
    
      // YZ-transpose within the same matrix from the 0-element towards the second half of the matrix
    timer_start("inv_yz_transpose");
    yz_transpose_in_place_inv(data, size, pSSize);                                                         // it's in data
    timer_stop("inv_yz_transpose");

      // out of place FFTs in Z-direction from the second half of the matrix towards the 0-element
    timer_start("inv_fft_on_z");
    bigfft_execC2R(plan->invPlanZ, (cufftComplex*)(data + half_pSSize), (cufftReal*)data );                // it's in data
    gpu_sync();
    timer_stop("inv_fft_on_z");

  }
  else {          //no zero padding in X- and Y direction (e.g. for Greens kernel computations)
      // out of place FFTs in Y-direction
    bigfft_execC2C(plan->planY, (cufftComplex*)data,  (cufftComplex*)data2, CUFFT_INVERSE);                // it's in data2
    gpu_sync();
    
      // YZ-transpose needs to be out of place.
    gpu_transposeYZ_complex(data2, data, N0, N2, N1*N3);                                                   // it's in data   

      // in place FFTs in Z-direction
    bigfft_execC2R(plan->invPlanZ, (cufftComplex*) data, (cufftReal*) data );                              // it's in data
    gpu_sync();
  }
  
  timer_start("inv_copy_to_unpad");
  gpu_copy_to_unpad(data, output, pSSize, size);                                                           // it's in output
  timer_stop("inv_copy_to_unpad");
 
//   timer_stop("gpu_plan3d_real_input_inverse_exec");
  
  return;
}



#ifdef __cplusplus
}
#endif