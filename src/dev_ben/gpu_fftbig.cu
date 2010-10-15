#include "gputil.h"
#include <cufft.h>
#include "gpu_transpose.h"
#include "gpu_transpose2.h"
#include "gpu_safe.h"
#include "gpu_fft.h"
#include "gpu_fftbig.h"
#include "gpu_conf.h"

#ifdef __cplusplus
extern "C" {
#endif

int MAXSIZE = 8*1024*1024;

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

  plan->Nbatch    = (int *) calloc(3, sizeof(int));
  plan->batch     = (int**) calloc(3, sizeof(int*));
  plan->batch_cum = (int**) calloc(3, sizeof(int*));

  int *Nbatch = plan->Nbatch;
  int **batch = plan->batch;

  init_batch_fft_big(plan, Z, size[X]*size[Y], plan->paddedStorageSize[Z]);
  for (int i=0; i<Nbatch[Z]; i++)
    printf("Z: Nbatch: %d, i: %d, batch: %d, el. in batch: %d, %d, batch_cum: %d\n", Nbatch[Z], i, batch[Z][i], batch[Z][i]*paddedSize[Z], batch[Z][i]*paddedStorageSize[Z], plan->batch_cum[Z][i]);
  printf("\n\n");
  init_batch_fft_big(plan, Y, paddedStorageSize[Z] * size[X] / 2, 2*plan->paddedSize[Y]);   //2* because we need the number of floats
  for (int i=0; i<Nbatch[Y]; i++)
    printf("Y: Nbatch: %d, i: %d, batch: %d, el. in batch: %d, batch_cum: %d\n", Nbatch[Y], i, batch[Y][i], batch[Y][i]*plan->paddedStorageSize[Y], plan->batch_cum[Y][i]);
  printf("\n\n");
  init_batch_fft_big(plan, X, paddedStorageSize[Z] * paddedSize[Y] / 2, 2*plan->paddedSize[X]);
  for (int i=0; i<Nbatch[X]; i++)
    printf("X: Nbatch: %d, i: %d, batch: %d, el. in batch: %d, batch_cum: %d\n", Nbatch[X], i, batch[X][i], batch[X][i]*plan->paddedStorageSize[X], plan->batch_cum[X][i]);
  printf("\n\n");
 
  // plan assignment for batch PlanZ -------------------------------------
  gpu_safefft( cufftPlan1d( &plan->fwPlanZ_1, plan->paddedSize[Z], CUFFT_R2C, batch[Z][0]) );
  gpu_safefft( cufftPlan1d( &plan->invPlanZ_1, plan->paddedSize[Z], CUFFT_C2R, batch[Z][0]) );
  if ( batch[Z][Nbatch[Z]-1] != batch[Z][0] ){
    gpu_safefft( cufftPlan1d( &plan->fwPlanZ_2, plan->paddedSize[Z], CUFFT_R2C, batch[Z][Nbatch[Z]-1]) );
    gpu_safefft( cufftPlan1d( &plan->invPlanZ_2, plan->paddedSize[Z], CUFFT_C2R, batch[Z][Nbatch[Z]-1]) );
  }

  plan->fwPlanZ  = (cufftHandle *) calloc(Nbatch[Z], sizeof(cufftHandle));
  plan->invPlanZ = (cufftHandle *) calloc(Nbatch[Z], sizeof(cufftHandle));
  
  for (int i=0; i<Nbatch[Z]; i++)
    if ( batch[Z][i] == batch[Z][0] ){
      plan->fwPlanZ[i] = plan->fwPlanZ_1;
      plan->invPlanZ[i] = plan->invPlanZ_1;
    }
    else{
      plan->fwPlanZ[i] = plan->fwPlanZ_2;
      plan->invPlanZ[i] = plan->invPlanZ_2;
    }
  //-----------------------------------------------------------------------
 

  // plan assignment for batch PlanY -------------------------------------
  gpu_safefft( cufftPlan1d( &plan->PlanY_1, plan->paddedSize[Y], CUFFT_C2C, batch[Y][0]) );
  if ( batch[Y][Nbatch[Y]-1] != batch[Y][0] ){
    gpu_safefft( cufftPlan1d( &plan->PlanY_2, plan->paddedSize[Y], CUFFT_C2C, batch[Y][Nbatch[Y]-1]) );
  }

  plan->planY  = (cufftHandle *) calloc(Nbatch[Y], sizeof(cufftHandle));
  
  for (int i=0; i<Nbatch[Y]; i++)
    if ( batch[Y][i] == batch[Y][0] )
      plan->planY[i] = plan->PlanY_1;
    else
      plan->planY[i] = plan->PlanY_2;

  //-----------------------------------------------------------------------


  // plan assignment for batch PlanX -------------------------------------
  gpu_safefft( cufftPlan1d( &plan->PlanX_1, plan->paddedSize[X], CUFFT_C2C, batch[X][0]) );
  if ( batch[X][Nbatch[X]-1] != batch[X][0] ){
    gpu_safefft( cufftPlan1d( &plan->PlanX_2, plan->paddedSize[X], CUFFT_C2C, batch[X][Nbatch[X]-1]) );
  }

  plan->planX  = (cufftHandle *) calloc(Nbatch[X], sizeof(cufftHandle));
  
  for (int i=0; i<Nbatch[X]; i++)
    if ( batch[X][i] == batch[X][0] )
      plan->planX[i] = plan->PlanX_1;
    else
      plan->planX[i] = plan->PlanX_2;
  //-----------------------------------------------------------------------

//   gpu_safefft( cufftPlan1d((&plan->PlanX_1), plan->paddedSize[X], CUFFT_C2C, paddedStorageSize[Z] * paddedSize[Y] / 2) );
//   gpu_safefft( cufftPlan1d((&plan->invPlanZ_1), plan->paddedSize[Z], CUFFT_C2R, size[X]*size[Y]) );
  
  plan->transp = new_gpu_array(plan->paddedStorageN);
  
  return plan;
}

void init_batch_fft_big(gpuFFT3dPlan_big *plan, int co, int Nffts, int size_fft){

  int K = get_factor_to_stride(size_fft);
  int max = MAXSIZE/size_fft/K;

  if (MAXSIZE < size_fft*K){
    printf("size %d does not fit the fft batch!", size_fft);
    abort();
  }

  int Nbatch = Nffts/K/max + (Nffts%(K*max) + K*max - 1)/(K*max);
  plan->Nbatch[co] = Nbatch;
  plan->batch[co]     = (int*) calloc(Nbatch, sizeof(int));
  plan->batch_cum[co] = (int*) calloc(Nbatch, sizeof(int));

  int left = Nffts;
  int cnt=0;
  while (left >= K*max){
    plan->batch[co][cnt] = K*max;
    left -=K*max;
    cnt++;
  }
  if (left!=0)
    plan->batch[co][cnt] = left;
    
  plan->batch_cum[co][0] = 0;
  for (int i=1; i< Nbatch; i++)
    plan->batch_cum[co][i] = plan->batch_cum[co][i-1] + plan->batch[co][i-1];

  return;
}


int get_factor_to_stride(int size_fft){

  int K = 1;
  int stride = gpu_stride_float();
  while (size_fft % (stride/K) !=0)
    K*=2;

  return (K);
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
    gpu_copy_to_pad(input, output, size, pSSize);

  
//  float* data = input;
  float* data = output;
  float* data2 = plan->transp; 

  if ( pSSize[X]!=size[X] || pSSize[Y]!=size[Y]){
      // out of place FFTs in Z-direction from the 0-element towards second half of the zeropadded matrix (out of place: no +2 on input!)
    for (int i=0; i<plan->Nbatch[Z]; i++){
      int index_in  = plan->batch_cum[Z][i]*plan->paddedSize[Z];
      int index_out = half_pSSize + plan->batch_cum[Z][i]*pSSize[Z];
      gpu_safefft( cufftExecR2C(plan->fwPlanZ[i], (cufftReal*)(data + index_in),  (cufftComplex*)(data + index_out) ) );     // it's in data
    }
    gpu_sync();

      // zero out the input data points at the start of the matrix
    gpu_zero(data, size[X]*size[Y]*pSSize[Z]);
    
      // YZ-transpose within the same matrix from the second half of the matrix towards the 0-element
    yz_transpose_in_place_fw(data, size, pSSize);                                                          // it's in data

    // in place FFTs in Y-direction
    for (int i=0; i<plan->Nbatch[Y]; i++){
      int index  = 2*plan->batch_cum[Y][i]*plan->paddedSize[Y];
      gpu_safefft( cufftExecC2C(plan->planY[i], (cufftComplex*) (data + index),  (cufftComplex*) (data + index), CUFFT_FORWARD) );       // it's in data 
    }
    gpu_sync();
  }
  
  else {          //no zero padding in X- and Y direction (e.g. for Greens kernel computations)
      // in place FFTs in Z-direction (there is no zero space to perform them out of place)
    for (int i=0; i<plan->Nbatch[Z]; i++){
      int index_in  = plan->batch_cum[Z][i]*plan->paddedSize[Z];
      int index_out = plan->batch_cum[Z][i]*plan->paddedSize[Z];
      gpu_safefft( cufftExecR2C(plan->fwPlanZ[i], (cufftReal*)(data + index_in),  (cufftComplex*)(data + index_out) ) );     // it's in data
    }
    gpu_sync();
    
      // YZ-transpose needs to be out of place.
    gpu_transposeYZ_complex(data, data2, N0, N1, N2*N3);                                                   // it's in data2
    
      // perform the FFTs in the Y-direction
    gpu_safefft( cufftExecC2C(plan->PlanY_1, (cufftComplex*)data2,  (cufftComplex*)data, CUFFT_FORWARD) );      // it's in data
    gpu_sync();
  }

  if(N0 > 1){    // not done for 2D transforms
      // XZ transpose still needs to be out of place
    gpu_transposeXZ_complex(data, data2, N0, N2, N1*N3);                                                   // it's in data2
 
    
      // out of place FFTs in X-direction
    for (int i=0; i<plan->Nbatch[X]; i++){
     int index  = 2*plan->batch_cum[X][i]*plan->paddedSize[X];
     gpu_safefft( cufftExecC2C(plan->planX[i], (cufftComplex*) (data2 + index),  (cufftComplex*) (output + index), CUFFT_FORWARD) );       // it's in data 
    }
    gpu_sync();
    
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
    for (int i=0; i<plan->Nbatch[X]; i++){
     int index  = 2*plan->batch_cum[X][i]*plan->paddedSize[X];
     gpu_safefft( cufftExecC2C(plan->planX[i], (cufftComplex*) (data + index),  (cufftComplex*) (data2 + index), CUFFT_INVERSE) );       // it's in data 
    }
    gpu_sync();

      // XZ transpose still needs to be out of place
    gpu_transposeXZ_complex(data2, data, N1, N2, N0*N3);                                                   // it's in data
  }
  
  if ( pSSize[X]!=size[X] || pSSize[Y]!=size[Y]){
      // in place FFTs in Y-direction
    for (int i=0; i<plan->Nbatch[Y]; i++){
      int index  = 2*plan->batch_cum[Y][i]*plan->paddedSize[Y];
      gpu_safefft( cufftExecC2C(plan->planY[i], (cufftComplex*) (data + index),  (cufftComplex*) (data + index), CUFFT_INVERSE) );       // it's in data 
    }
    gpu_sync();
    
    
      // YZ-transpose within the same matrix from the 0-element towards the second half of the matrix
    yz_transpose_in_place_inv(data, size, pSSize);                                                          // it's in data

      // out of place FFTs in Z-direction from the second half of the matrix towards the 0-element
    for (int i=0; i<plan->Nbatch[Z]; i++){
      int index_in  = half_pSSize + plan->batch_cum[Z][i]*pSSize[Z];
      int index_out = plan->batch_cum[Z][i]*plan->paddedSize[Z];
      gpu_safefft( cufftExecC2R(plan->invPlanZ[i], (cufftComplex*)(data + index_in),  (cufftReal*)(data + index_out) ) );     // it's in data
    }
    gpu_sync();

  }
  else {          //no zero padding in X- and Y direction (e.g. for Greens kernel computations)
      // out of place FFTs in Y-direction
    gpu_safefft( cufftExecC2C(plan->planY[0], (cufftComplex*)data,  (cufftComplex*)data2, CUFFT_INVERSE) );       // it's in data
    gpu_sync();
    
      // YZ-transpose needs to be out of place.
    gpu_transposeYZ_complex(data2, data, N0, N2, N1*N3);                                                    // it's in data2   

      // in place FFTs in Z-direction
    gpu_safefft( cufftExecC2R(plan->invPlanZ_1, (cufftComplex*) data, (cufftReal*) data ));                      // it's in data
    gpu_sync();
  }
  
  gpu_copy_to_unpad(data, output, pSSize, size);
 
//   timer_stop("gpu_plan3d_real_input_inverse_exec");
  
  return;
}



#ifdef __cplusplus
}
#endif