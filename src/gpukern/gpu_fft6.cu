#include "gpu_fft6.h"
#include "timer.h"

#ifdef __cplusplus
extern "C" {
#endif

//#define BLOCKSIZE 32
#define BLOCKSIZE 16

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
    gpu_transpose_complex_YZ(data, data2, N1, N2*N3, N0);                                                   // it's in data2
    
      // perform the FFTs in the Y-direction
    bigfft_execC2C(plan->planY, (cufftComplex*)data2,  (cufftComplex*)data, CUFFT_FORWARD);                // it's in data
    gpu_sync();
  }

  if(N0 > 1){    // not done for 2D transforms
      // XZ transpose still needs to be out of place
    timer_start("fw_xz_transpose");
//     gpu_transposeXZ_complex(data, data2, N0, N2, N1*N3);                                                   // it's in data2
    gpu_transpose_complex_XZ(data, data2, pSSize[0], pSSize[1], pSSize[2]);
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
//     gpu_transposeXZ_complex(data2, data, N1, N2, N0*N3);                                                   // it's in data
    gpu_transpose_complex_XZ(data2, data, pSSize[2]/2, pSSize[1], pSSize[0]*2);
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
    gpu_transpose_complex_YZ(data2, data, N2, N1*N3, N0);                                                   // it's in data   

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
  int stride1 = size[Y]*pSSize[Z]/2;
  int stride2 = pSSize[Y]*pSSize[Z]/2;

  if (size[X]!=pSSize[X]){
      // transpose all planes out of place
    gpu_transpose_complex_offset2(data + offset, data, size[Y], pSSize[Z], 0, pSSize[Y]-size[Y], size[X], stride1, stride2);
    gpu_sync();
  }
  else{     //padding in the y-direction
      // transpose all but the last plane out of place, is only partially parallel
    int done = 0;
    while ( done<(size[X]-1) ){
      int N = (size[X]-done)/2;
      int ind1 = offset + done*size[Y]*pSSize[Z];
      int ind2 = done*pSSize_YZ;
      gpu_transpose_complex_offset2(data + ind1, data + ind2, size[Y], pSSize[Z], 0, pSSize[Y]-size[Y], N, stride1, stride2);
      done += N;
      gpu_sync();
    }
    gpu_transpose_complex_in_plane_fw(data + (size[X]-1)*pSSize_YZ, size[Y], pSSize[Z]);
  }
  
  return;
}

void yz_transpose_in_place_inv(float *data, int *size, int *pSSize){

  int offset = pSSize[X]*pSSize[Y]*pSSize[Z]/2;    //start of second half
  int pSSize_YZ = pSSize[Y]*pSSize[Z];
  int stride1 = pSSize[Y]*pSSize[Z]/2;
  int stride2 = size[Y]*pSSize[Z]/2;

  if (size[X]!=pSSize[X]){
      // transpose all planes out of place
    gpu_transpose_complex_offset2(data, data + offset, pSSize[Z]/2, 2*size[Y], pSSize[Y]-size[Y], 0, size[X], stride1, stride2);
    gpu_sync();
  }
  else{
      // last plane needs to transposed in plane
    gpu_transpose_complex_in_plane_inv(data + (size[X]-1)*pSSize_YZ, pSSize[Z]/2, 2*size[Y]);
      // transpose all but the last plane out of place: is partly be parallellized
    int left = size[X]-1;
    while ( left>0 ){
      int N = size[X]-left;
      left -= N;
      int ind1 = (left)*pSSize_YZ;
      int ind2 = offset + (left)*size[Y]*pSSize[Z];
      gpu_transpose_complex_offset2(data + ind1, data + ind2, pSSize[Z]/2, 2*size[Y], pSSize[Y]-size[Y], 0, N, stride1, stride2);
      gpu_sync();
    }
  }
  
  return;
}



// functions for copying to and from padded matrix ****************************************************

/// @internal Does padding and unpadding, not necessarily by a factor 2
__global__ void _gpu_copy_pad(int N, float* source, float* dest, 
                               int S1, int S2,                  ///< source sizes Y and Z
                               int D1, int D2                   ///< destination size Y and Z
                               ){

  ///x-index is always running the fastest!

  int k = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  if(j<S1 && k<S2)
    for (int i=0; i<N; i++)
      dest[(i*D1 + j)*D2 + k] = source[(i*S1 + j)*S2 + k];

  return;
}

void gpu_copy_to_pad(float* source, float* dest, int *unpad_size, int *pad_size){          //for padding of the tensor, 2d and 3d applicable
  
  int S0 = unpad_size[X];
  int S1 = unpad_size[Y];
  int S2 = unpad_size[Z];

  
  dim3 gridSize(divUp(S2, BLOCKSIZE), divUp(S1, BLOCKSIZE), 1);
  dim3 blockSize(BLOCKSIZE, BLOCKSIZE, 1);
  check3dconf(gridSize, blockSize);
  
//   printf("S: %d, %d, D: %d, %d\n", S1, S2, S1, pad_size[Z]-2);

  if ( pad_size[X]!=unpad_size[X] || pad_size[Y]!=unpad_size[Y])
    _gpu_copy_pad<<<gridSize, blockSize>>>(S0, source, dest, S1, S2, S1, pad_size[Z]-2);      // for out of place forward FFTs in z-direction, contiguous data arrays
  else
    _gpu_copy_pad<<<gridSize, blockSize>>>(S0, source, dest, S1, S2, S1, pad_size[Z]);        // for in place forward FFTs in z-direction, contiguous data arrays

  gpu_sync();
  
  return;
}

void gpu_copy_to_unpad(float* source, float* dest, int *pad_size, int *unpad_size){        //for unpadding of the tensor, 2d and 3d applicable
  
  int D0 = unpad_size[X];
  int D1 = unpad_size[Y];
  int D2 = unpad_size[Z];

  dim3 gridSize(divUp(D2, BLOCKSIZE), divUp(D1, BLOCKSIZE), 1);
  dim3 blockSize(BLOCKSIZE, BLOCKSIZE, 1);
  check3dconf(gridSize, blockSize);

  if ( pad_size[X]!=unpad_size[X] || pad_size[Y]!=unpad_size[Y])
    _gpu_copy_pad<<<gridSize, blockSize>>>(D0, source, dest, D1,  pad_size[Z]-2, D1, D2);       // for out of place inverse FFTs in z-direction, contiguous data arrays
  else
    _gpu_copy_pad<<<gridSize, blockSize>>>(D0, source, dest, D1,  pad_size[Z], D1, D2);         // for in place inverse FFTs in z-direction, contiguous data arrays
    
  gpu_sync();
  
  return;
}


void delete_FFT3dPlan(gpuFFT3dPlan* kernel_plan){

  return;
}


// __global__ void _gpu_copy_pad(int N, float* source, float* dest, 
//                                int S1, int S2,                  ///< source sizes Y and Z
//                                int D1, int D2                   ///< destination size Y and Z
//                                ){
// 
//   ///@todo check timing with x<->y
//   int i = blockIdx.x/N;
//   int K = blockIdx.x%N;
//   int k = K * blockDim.x + threadIdx.x;
//   int j = blockIdx.y * blockDim.y + threadIdx.y;
//   
//   if(j<S1 && k<S2)
//     dest[(i*D1 + j)*D2 + k] = source[(i*S1 + j)*S2 + k];
// 
//   return;
// }
// 
// 
// void gpu_copy_to_pad(float* source, float* dest, int *unpad_size, int *pad_size){          //for padding of the tensor, 2d and 3d applicable
//   
//   int S0 = unpad_size[X];
//   int S1 = unpad_size[Y];
//   int S2 = unpad_size[Z];
// 
//   
//   dim3 gridSize(S0*divUp(S2, BLOCKSIZE), divUp(S1, BLOCKSIZE), 1);
//   dim3 blockSize(BLOCKSIZE, BLOCKSIZE, 1);
//   check3dconf(gridSize, blockSize);
// 
//   if ( pad_size[X]!=unpad_size[X] || pad_size[Y]!=unpad_size[Y])
//     _gpu_copy_pad<<<gridSize, blockSize>>>(divUp(S2, BLOCKSIZE), source, dest, S1, S2, S1, pad_size[Z]-2);      // for out of place forward FFTs in z-direction, contiguous data arrays
//   else
//     _gpu_copy_pad<<<gridSize, blockSize>>>(divUp(S2, BLOCKSIZE), source, dest, S1, S2, S1, pad_size[Z]);        // for in place forward FFTs in z-direction, contiguous data arrays
// 
//   gpu_sync();
//   
//   return;
// }
// 
// void gpu_copy_to_unpad(float* source, float* dest, int *pad_size, int *unpad_size){        //for unpadding of the tensor, 2d and 3d applicable
//   
//   int D0 = unpad_size[X];
//   int D1 = unpad_size[Y];
//   int D2 = unpad_size[Z];
// 
//   dim3 gridSize(D0*divUp(D2, BLOCKSIZE), divUp(D1, BLOCKSIZE), 1);
//   dim3 blockSize(BLOCKSIZE, BLOCKSIZE, 1);
//   check3dconf(gridSize, blockSize);
// 
//   if ( pad_size[X]!=unpad_size[X] || pad_size[Y]!=unpad_size[Y])
//     _gpu_copy_pad<<<gridSize, blockSize>>>(divUp(D2, BLOCKSIZE), source, dest, D1,  pad_size[Z]-2, D1, D2);       // for out of place inverse FFTs in z-direction, contiguous data arrays
//   else
//     _gpu_copy_pad<<<gridSize, blockSize>>>(divUp(D2, BLOCKSIZE), source, dest, D1,  pad_size[Z], D1, D2);         // for in place inverse FFTs in z-direction, contiguous data arrays
//     
//   gpu_sync();
//   
//   return;
// }
// 
// 
// 
// 
// 
// __global__ void _gpu_copy_pad3(int N2, int N1, int N, float* source, float* dest, 
//                                int S1, int S2,                  ///< source sizes Y and Z
//                                int D1, int D2                   ///< destination size Y and Z
//                                ){
// 
//   ///@todo x is fastest running index
//   int i = blockIdx.x/N2*N + blockIdx.y/N1;
//   int k = blockIdx.x%N2 * blockDim.x + threadIdx.x;
//   int j = blockIdx.y%N1 * blockDim.y + threadIdx.y;
//   
//   if(j<S1 && k<S2)
//     dest[(i*D1 + j)*D2 + k] = source[(i*S1 + j)*S2 + k];
// 
//   return;
// }
// 
// 
// void gpu_copy_to_pad3(float* source, float* dest, int *unpad_size, int *pad_size){          //for padding of the tensor, 2d and 3d applicable
//   
// //   int S0 = unpad_size[X];
//   int S0_1, S0_2;
//   int S1 = unpad_size[Y];
//   int S2 = unpad_size[Z];
// 
//   split_int (unpad_size[X], &S0_1, &S0_2);
//   dim3 gridSize(S0_2*divUp(S2, BLOCKSIZE), S0_1*divUp(S1, BLOCKSIZE), 1);
//   dim3 blockSize(BLOCKSIZE, BLOCKSIZE, 1);
//   check3dconf(gridSize, blockSize);
// 
//   if ( pad_size[X]!=unpad_size[X] || pad_size[Y]!=unpad_size[Y])
//     _gpu_copy_pad3<<<gridSize, blockSize>>>(divUp(S2, BLOCKSIZE), divUp(S1, BLOCKSIZE), S0_1, source, dest, S1, S2, S1, pad_size[Z]-2);      // for out of place forward FFTs in z-direction, contiguous data arrays
//   else
//     _gpu_copy_pad3<<<gridSize, blockSize>>>(divUp(S2, BLOCKSIZE), divUp(S1, BLOCKSIZE), S0_1, source, dest, S1, S2, S1, pad_size[Z]);        // for in place forward FFTs in z-direction, contiguous data arrays
// 
//   gpu_sync();
//   
//   return;
// }
// 
// void gpu_copy_to_unpad3(float* source, float* dest, int *pad_size, int *unpad_size){        //for unpadding of the tensor, 2d and 3d applicable
//   
// //   int D0 = unpad_size[X];
//   int D0_1, D0_2;
//   int D1 = unpad_size[Y];
//   int D2 = unpad_size[Z];
//   
//   split_int (unpad_size[X], &D0_1, &D0_2);
// 
//   dim3 gridSize(D0_2*divUp(D2, BLOCKSIZE), D0_1*divUp(D1, BLOCKSIZE), 1);
//   dim3 blockSize(BLOCKSIZE, BLOCKSIZE, 1);
//   check3dconf(gridSize, blockSize);
// 
//   if ( pad_size[X]!=unpad_size[X] || pad_size[Y]!=unpad_size[Y])
//     _gpu_copy_pad3<<<gridSize, blockSize>>>(divUp(D2, BLOCKSIZE), divUp(D1, BLOCKSIZE), D0_1, source, dest, D1,  pad_size[Z]-2, D1, D2);       // for out of place inverse FFTs in z-direction, contiguous data arrays
//   else
//     _gpu_copy_pad3<<<gridSize, blockSize>>>(divUp(D2, BLOCKSIZE), divUp(D1, BLOCKSIZE), D0_1, source, dest, D1,  pad_size[Z], D1, D2);         // for in place inverse FFTs in z-direction, contiguous data arrays
//     
//   gpu_sync();
//   
//   return;
// }
// 
// 
// void split_int (int N, int *N1, int *N2){
// 
//   int sq = (int) sqrt(N);
//   
//   while(sq>=1){
//     if (N%sq==0){
//       *N1 = N/sq;
//       *N2 = sq;
//       break;
//     }
//     sq -=1;
//   }
// 
//   return;
// }
// 
// 





















/// @internal Does padding and unpadding, not necessarily by a factor 2
__global__ void _gpu_copy_pad2(int N0, float* source, float* dest, 
                               int S1, int S2,                  ///< source sizes Y and Z
                               int D1, int D2                   ///< destination size Y and Z
                               ){

  ///x-index is always running the fastest.
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int k = blockIdx.x * blockDim.x + threadIdx.x;

  if(j<S1 && k<S2){
    for (int i=0; i<N0; i++)
      dest[(i*D1 + j)*D2 + k] = source[(i*S1 + j)*S2 + k];
  }
 return;
}


void gpu_copy_to_pad2(float* source, float* dest, int *unpad_size, int *pad_size){          //for padding of the tensor, 2d and 3d applicable
  
  int S0 = unpad_size[0];
  int S1 = unpad_size[1];
  int S2 = unpad_size[2];

  
  dim3 gridSize(divUp(S2, BLOCKSIZE), divUp(S1, BLOCKSIZE), 1);
  dim3 blockSize(BLOCKSIZE, BLOCKSIZE, 1);
  check3dconf(gridSize, blockSize);

  if ( pad_size[0]!=unpad_size[0] || pad_size[1]!=unpad_size[1])
    _gpu_copy_pad2<<<gridSize, blockSize>>>(S0, source, dest, S1, S2, S1, pad_size[2]-2);      // for out of place forward FFTs in z-direction, contiguous data arrays
  else
    _gpu_copy_pad2<<<gridSize, blockSize>>>(S0, source, dest, S1, S2, S1, pad_size[2]);        // for in place forward FFTs in z-direction, contiguous data arrays
  gpu_sync();
  
  return;
}


void gpu_copy_to_unpad2(float* source, float* dest, int *pad_size, int *unpad_size){        //for unpadding of the tensor, 2d and 3d applicable
  
  int D0 = unpad_size[X];
  int D1 = unpad_size[Y];
  int D2 = unpad_size[Z];

  dim3 gridSize(divUp(D2, BLOCKSIZE), divUp(D1, BLOCKSIZE), 1);
  dim3 blockSize(BLOCKSIZE, BLOCKSIZE, 1);
  check3dconf(gridSize, blockSize);

  if ( pad_size[X]!=unpad_size[X] || pad_size[Y]!=unpad_size[Y])
    _gpu_copy_pad2<<<gridSize, blockSize>>>(D0, source, dest, D1,  pad_size[Z]-2, D1, D2);       // for out of place inverse FFTs in z-direction, contiguous data arrays
  else
    _gpu_copy_pad2<<<gridSize, blockSize>>>(D0, source, dest, D1,  pad_size[Z], D1, D2);         // for in place inverse FFTs in z-direction, contiguous data arrays
    
  gpu_sync();
  
  return;
}
// ****************************************************************************************************

#ifdef __cplusplus
}
#endif