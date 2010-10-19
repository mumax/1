#include "gpu_fftbig.h"

#ifdef __cplusplus
extern "C" {
#endif

int MAXSIZE = 1024;

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
  init_batch_fft_big(plan, Y, paddedStorageSize[Z] * size[X] / 2, plan->paddedSize[Y]);
  init_batch_fft_big(plan, X, paddedStorageSize[Z] * paddedSize[Y] / 2, plan->paddedSize[X]);
  printf("%d, %d, %d\n", Nbatch[X], batch[X][0], plan->batch_cum[X][0]);
  printf("%d, %d, %d\n", Nbatch[Y], batch[Y][0], plan->batch_cum[Y][0]);
  printf("%d, %d, %d\n", Nbatch[Z], batch[Z][0], plan->batch_cum[Z][0]);
  
  plan->fwPlanZ  = (cufftHandle *) calloc(Nbatch[Z], sizeof(cufftHandle));
  gpu_safefft( cufftPlan1d( &plan->fwPlanZ_1, plan->paddedSize[Z], CUFFT_R2C, batch[Z][0]) );
  if ( batch[Z][Nbatch[Z]-1] != batch[Z][0] )
    gpu_safefft( cufftPlan1d( &plan->fwPlanZ_2, plan->paddedSize[Z], CUFFT_R2C, batch[Z][Nbatch[Z]-1]) );
/*  else
    plan->fwPlanZ_2 = NULL;*/
  for (int i=0; i<Nbatch[Z]; i++)
    if ( batch[Z][i] == batch[Z][0] )
      plan->fwPlanZ[i] = plan->fwPlanZ_1;
    else
      plan->fwPlanZ[i] = plan->fwPlanZ_2;
  printf("ha  ll\n");
    
  for (int i=0; i<plan->Nbatch[Z]; i++)
    gpu_safefft( cufftPlan1d( &plan->fwPlanZ[i], plan->paddedSize[Z], CUFFT_R2C, plan->batch[Z][i]) );

  plan->planY  = (cufftHandle *) calloc(plan->Nbatch[Y], sizeof(cufftHandle));
  for (int i=0; i<plan->Nbatch[Y]; i++)
    gpu_safefft( cufftPlan1d( &plan->planY[i], plan->paddedSize[Y], CUFFT_C2C, plan->batch[Y][i]) );
  
  plan->planX  = (cufftHandle *) calloc(plan->Nbatch[X], sizeof(cufftHandle));
  for (int i=0; i<plan->Nbatch[X]; i++)
    gpu_safefft( cufftPlan1d( &plan->planX[i], plan->paddedSize[X], CUFFT_C2C, plan->batch[X][i]) );
  
  plan->invPlanZ  = (cufftHandle *) calloc(plan->Nbatch[Z], sizeof(cufftHandle));
  for (int i=0; i<plan->Nbatch[Z]; i++)
    gpu_safefft( cufftPlan1d( &plan->invPlanZ[i], plan->paddedSize[Z], CUFFT_C2R, plan->batch[Z][i]) );

  gpu_safefft( cufftPlan1d((&plan->planY[0]), plan->paddedSize[Y], CUFFT_C2C, paddedStorageSize[Z] * size[X] / 2) );          // IMPORTANT: the /2 is necessary because the complex transforms have only half the amount of elements (the elements are now complex numbers)
  gpu_safefft( cufftPlan1d((&plan->planX[0]), plan->paddedSize[X], CUFFT_C2C, paddedStorageSize[Z] * paddedSize[Y] / 2) );
  gpu_safefft( cufftPlan1d((&plan->invPlanZ[0]), plan->paddedSize[Z], CUFFT_C2R, size[X]*size[Y]) );
  
  plan->transp = new_gpu_array(plan->paddedStorageN);
  
  return plan;
}

void init_batch_fft_big(gpuFFT3dPlan_big *plan, int co, int Nffts, int size_fft){

  int max = MAXSIZE/size_fft;

  int Nbatch = Nffts/max + (Nffts%max + max - 1)/max;
  plan->Nbatch[co] = Nbatch;
  plan->batch[co]     = (int*) calloc(Nbatch, sizeof(int));
  plan->batch_cum[co] = (int*) calloc(Nbatch, sizeof(int));
  
  for (int i=0; i< Nbatch; i++){
    plan->batch[co][i] = Nffts/ Nbatch;
    if ( (Nffts % Nbatch) > i)
      plan->batch[co][i]++;
  }
  
  plan->batch_cum[co][0] = 0;
  for (int i=1; i< Nbatch; i++)
    plan->batch_cum[co][i] = plan->batch_cum[co][i-1] + plan->batch[co][i];

  printf("%d, %d\n", plan->Nbatch[co], plan->batch_cum[co][0]);

  return;
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
    
      // out of place FFTs in Z-direction from the 0-element towards second half of the zeropadded matrix (out of place: no +2 on input!)
//     gpu_safefft( cufftExecR2C(plan->fwPlanZ, (cufftReal*)data,  (cufftComplex*) (data + half_pSSize) ) );     // it's in data
    gpu_sync();
      // zero out the input data points at the start of the matrix
    gpu_zero(data, size[X]*size[Y]*pSSize[Z]);
    
      // YZ-transpose within the same matrix from the second half of the matrix towards the 0-element
    yz_transpose_in_place_fw(data, size, pSSize);                                                          // it's in data
    
      // in place FFTs in Y-direction
    gpu_safefft( cufftExecC2C(plan->planY[0], (cufftComplex*)data,  (cufftComplex*)data, CUFFT_FORWARD) );       // it's in data 
    gpu_sync();
  }
  
  else {          //no zero padding in X- and Y direction (e.g. for Greens kernel computations)
      // in place FFTs in Z-direction (there is no zero space to perform them out of place)
//     gpu_safefft( cufftExecR2C(plan->fwPlanZ[0], (cufftReal*)data,  (cufftComplex*) data ) );                     // it's in data
    for (int i=0; i<plan->Nbatch[Z]; i++){
      int index_in  = plan->batch_cum[Z][i]*plan->paddedSize[Z];
      int index_out = plan->batch_cum[Z][i]*pSSize[Z];
      gpu_safefft( cufftExecR2C(plan->fwPlanZ[i], (cufftReal*)(data + index_in),  (cufftComplex*)(data + index_out) ) );     // it's in data
    }
    gpu_sync();
    
      // YZ-transpose needs to be out of place.
    gpu_transposeYZ_complex(data, data2, N0, N1, N2*N3);                                                   // it's in data2
    
      // perform the FFTs in the Y-direction
    gpu_safefft( cufftExecC2C(plan->planY[0], (cufftComplex*)data2,  (cufftComplex*)data, CUFFT_FORWARD) );      // it's in data
    gpu_sync();
  }

  if(N0 > 1){    // not done for 2D transforms
      // XZ transpose still needs to be out of place
    gpu_transposeXZ_complex(data, data2, N0, N2, N1*N3);                                                   // it's in data2
    
      // in place FFTs in X-direction
    gpu_safefft( cufftExecC2C(plan->planX[0], (cufftComplex*)data2,  (cufftComplex*)output, CUFFT_FORWARD) );    // it's in data
    gpu_sync();
  }

//   timer_stop("gpu_plan3d_real_input_forward_exec");
  
  return;
}


#ifdef __cplusplus
}
#endif