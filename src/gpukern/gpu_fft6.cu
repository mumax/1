#include "gpu_fft6.h"
#include "timer.h"

#ifdef __cplusplus
extern "C" {
#endif


gpuFFT3dPlan* new_gpuFFT3dPlan_padded(int* size, int* paddedSize){

  int N0 = size[X];
  int N1 = size[Y];
  int N2 = size[Z];
  
  assert(paddedSize[X] > 0);
  assert(paddedSize[Y] > 1);
  assert(paddedSize[Z] > 1);
  
  gpuFFT3dPlan* plan = (gpuFFT3dPlan*)malloc(sizeof(gpuFFT3dPlan));
  
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
  if (N0>1)
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
  if (N0>1)
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
  if (N0>1)
    for (int i=0; i<plan->planX->Nbatch; i++)
      printf("X: Nbatch: %d, i: %d, batch: %d, batch_index_in: %d, batch_index_out: %d\n", plan->planX->Nbatch, i, plan->planX->batch[i], plan->planX->batch_index_in[i], plan->planX->batch_index_out[i]);
  printf("\n");  
  
  plan->transp = new_gpu_array(plan->paddedStorageN);
  
  return plan;
}


void gpuFFT3dPlan_forward(gpuFFT3dPlan* plan, float* input, float* output){

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
    gpu_copy_to_pad2(input, output, size, pSSize);
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
//     xz_transpose_out_of_place_fw(data, data2, pSSize)
    timer_stop("fw_xz_transpose");
 
    
      // out of place FFTs in X-direction
    timer_start("fw_fft_on_x");
    bigfft_execC2C(plan->planX, (cufftComplex*)data2, (cufftComplex*)output, CUFFT_FORWARD);               // it's in output
    gpu_sync();
    timer_stop("fw_fft_on_x");
    
  }

  return;
}

void gpuFFT3dPlan_inverse(gpuFFT3dPlan* plan, float* input, float* output){
  
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
  gpu_copy_to_unpad2(data, output, pSSize, size);                                                           // it's in output
  timer_stop("inv_copy_to_unpad");
 
 
  return;
}

int gpuFFT3dPlan_normalization(gpuFFT3dPlan* plan){
  return plan->paddedSize[X] * plan->paddedSize[Y] * plan->paddedSize[Z];
}


void yz_transpose_in_place_fw(float *data, int *size, int *pSSize){

  int offset = pSSize[X]*pSSize[Y]*pSSize[Z]/2;    //start of second half
  int pSSize_YZ = pSSize[Y]*pSSize[Z];

  if (size[X]!=pSSize[X]){
    for (int i=0; i<size[X]; i++){       // transpose each plane out of place: can be parallellized
      int ind1 = offset + i*size[Y]*pSSize[Z];
      int ind2 = i*pSSize_YZ;
      gpu_transpose_complex_offset(data + ind1, data + ind2, size[Y], pSSize[Z], 0, pSSize[Y]-size[Y]);
    }
    gpu_sync();
    gpu_zero(data + offset, offset);     // possible to delete values in gpu_transpose_complex
  }
  else{     //padding in the y-direction
    for (int i=0; i<size[X]-1; i++){       // transpose all but the last plane out of place: can only partly be parallellized
      int ind1 = offset + i*size[Y]*pSSize[Z];
      int ind2 = i*pSSize_YZ;
      gpu_transpose_complex_offset(data + ind1, data + ind2, size[Y], pSSize[Z], 0, pSSize[Y]-size[Y]);
      gpu_zero(data + offset + i*size[Y]*pSSize[Z], size[Y]*pSSize[Z]);     // deletable
    }
    gpu_transpose_complex_in_plane_fw(data + (size[X]-1)*pSSize_YZ, size[Y], pSSize[Z]);
  }
  
  return;
}

void yz_transpose_in_place_inv(float *data, int *size, int *pSSize){

  int offset = pSSize[X]*pSSize[Y]*pSSize[Z]/2;    //start of second half
  int pSSize_YZ = pSSize[Y]*pSSize[Z];

  if (size[X]!=pSSize[X])
      // transpose each plane out of place: can be parallellized
    for (int i=0; i<size[X]; i++){
      int ind1 = i*pSSize_YZ;
      int ind2 = offset + i*size[Y]*pSSize[Z];
      gpu_transpose_complex_offset(data + ind1, data + ind2, pSSize[Z]/2, 2*size[Y], pSSize[Y]-size[Y], 0);
    }
  else{
      // last plane needs to transposed in plane
    gpu_transpose_complex_in_plane_inv(data + (size[X]-1)*pSSize_YZ, pSSize[Z]/2, 2*size[Y]);
      // transpose all but the last plane out of place: can only partly be parallellized
    for (int i=0; i<size[X]-1; i++){
      int ind1 = i*pSSize_YZ;
      int ind2 = offset + i*size[Y]*pSSize[Z];
      gpu_transpose_complex_offset(data + ind1, data + ind2, pSSize[Z]/2, 2*size[Y], pSSize[Y]-size[Y], 0);

    }
  }
  
  return;
}



void xz_transpose_out_of_place_fw(float *input, float *output, int *pSSize){

  for (int j=0; j<pSSize[Y]; j++){       // transpose each plane out of place
    int ind = j*pSSize[Z];
    gpu_transpose_complex_XZ(input + ind, output + ind, pSSize[X], pSSize[Y], pSSize[Z]);
  }
  gpu_sync();
  
  return;
}




// functions for copying to and from padded matrix ****************************************************

/// @internal Does padding and unpadding, not necessarily by a factor 2
__global__ void _gpu_copy_pad2(int i, float* source, float* dest, 
                               int S1, int S2,                  ///< source sizes Y and Z
                               int D1, int D2                   ///< destination size Y and Z
                               ){

  ///@todo check timing with x<->y
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  int k = blockIdx.y * blockDim.y + threadIdx.y;

  if(j<S1 && k<S2){
    dest[(i*D1 + j)*D2 + k] = source[(i*S1 + j)*S2 + k];
  }
 return;
}

#define BLOCKSIZE 16

void gpu_copy_to_pad2(float* source, float* dest, int *unpad_size, int *pad_size){          //for padding of the tensor, 2d and 3d applicable
  
  int S0 = unpad_size[0];
  int S1 = unpad_size[1];
  int S2 = unpad_size[2];

  
  dim3 gridSize(divUp(S1, BLOCKSIZE), divUp(S2, BLOCKSIZE), 1);
  dim3 blockSize(BLOCKSIZE, BLOCKSIZE, 1);
  check3dconf(gridSize, blockSize);

  for (int i=0; i<S0; i++){
    if ( pad_size[0]!=unpad_size[0] || pad_size[1]!=unpad_size[1])
      _gpu_copy_pad2<<<gridSize, blockSize>>>(i, source, dest, S1, S2, S1, pad_size[2]-2);      // for out of place forward FFTs in z-direction, contiguous data arrays
    else
      _gpu_copy_pad2<<<gridSize, blockSize>>>(i, source, dest, S1, S2, S1, pad_size[2]);        // for in place forward FFTs in z-direction, contiguous data arrays
  }
  gpu_sync();
  
  return;
}


void gpu_copy_to_unpad2(float* source, float* dest, int *pad_size, int *unpad_size){        //for unpadding of the tensor, 2d and 3d applicable
  
  int D0 = unpad_size[X];
  int D1 = unpad_size[Y];
  int D2 = unpad_size[Z];

  dim3 gridSize(divUp(D1, BLOCKSIZE), divUp(D2, BLOCKSIZE), 1);
  dim3 blockSize(BLOCKSIZE, BLOCKSIZE, 1);
  check3dconf(gridSize, blockSize);

  for (int i=0; i<D0; i++){
    if ( pad_size[X]!=unpad_size[X] || pad_size[Y]!=unpad_size[Y])
      _gpu_copy_pad2<<<gridSize, blockSize>>>(i, source, dest, D1,  pad_size[Z]-2, D1, D2);       // for out of place inverse FFTs in z-direction, contiguous data arrays
    else
      _gpu_copy_pad2<<<gridSize, blockSize>>>(i, source, dest, D1,  pad_size[Z], D1, D2);         // for in place inverse FFTs in z-direction, contiguous data arrays
  }
    
  gpu_sync();
  
  return;
}
// ****************************************************************************************************

#ifdef __cplusplus
}
#endif