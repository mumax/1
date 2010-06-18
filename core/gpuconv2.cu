#include "gpufft2.h"
#include "gpuconv2.h"
#include "gputil.h"
#include "timer.h"
#include <stdio.h>
#include <assert.h>

#ifdef __cplusplus
extern "C" {
#endif

//_____________________________________________________________________________________________ copy/pad

/// @internal Does padding and unpadding, not necessarily by a factor 2
__global__ void _gpuconv2_copy_pad(float* source, float* dest, 
                                   int S1, int S2,                  ///< source sizes Y and Z
                                   int D1, int D2                   ///< destination size Y and Z
                                   ){
  int i = blockIdx.x;
  int j = blockIdx.y;
  int k = threadIdx.x;

  dest[(i*D1 + j)*D2 + k] = source[(i*S1 + j)*S2 + k];
}


void gpu_copy_pad(tensor* source, tensor* dest){
  
  assert(source->rank == 3);
  assert(  dest->rank == 3);
  
  // source must not be larger than dest
  for(int i=0; i<3; i++){
    assert(source->size[i] <= dest->size[i]);
  }
  
  int S0 = source->size[X];
  int S1 = source->size[Y];
  int S2 = source->size[Z];

  dim3 gridSize(S0, S1, 1); ///@todo generalize!
  dim3 blockSize(S2, 1, 1);
  gpu_checkconf(gridSize, blockSize);

  _gpuconv2_copy_pad<<<gridSize, blockSize>>>(source->list, dest->list, S1, S2, dest->size[1], dest->size[2]);
  cudaThreadSynchronize();
}

void gpu_copy_unpad(tensor* source, tensor* dest){
  
  assert(source->rank == 3);
  assert(  dest->rank == 3);
  
  // dest must not be larger than source
  for(int i=0; i<3; i++){
    assert(source->size[i] >= dest->size[i]);
  }
  
  int D0 = dest->size[X];
  int D1 = dest->size[Y];
  int D2 = dest->size[Z];

  dim3 gridSize(D0, D1, 1); ///@todo generalize!
  dim3 blockSize(D2, 1, 1);
  gpu_checkconf(gridSize, blockSize);

  _gpuconv2_copy_pad<<<gridSize, blockSize>>>(source->list, dest->list, source->size[1], source->size[2], D1, D2);
  cudaThreadSynchronize();
}

//_____________________________________________________________________________________________ kernel multiplication

/**
 * @internal 
 * Kernel is in interleaved complex format (imaginary part is zero and not read, but still stored),
 * and assummed symmetric.
 * The multiplication is in-place, fftMi is overwritten by fftHi
 */
__global__ void _gpu_kernel_mul_complex_inplace_symm(float* fftMx,  float* fftMy,  float* fftMz, 
                                                     float* fftKxx, float* fftKyy, float* fftKzz,
                                                     float* fftKyz, float* fftKxz, float* fftKxy){
  
  int e = 2 * ((blockIdx.x * blockDim.x) + threadIdx.x);
  
  // we some shared memory here, which saves an "8N" buffer in the global memory
  ///@todo coalescale read/writes
  float reMx = fftMx[e  ];
  float imMx = fftMx[e+1];
  
  float reMy = fftMy[e  ];
  float imMy = fftMy[e+1];
  
  float reMz = fftMz[e  ];
  float imMz = fftMz[e+1];
 
  float Kxx = fftKxx[e];
  float Kyy = fftKyy[e];
  float Kzz = fftKzz[e];
  
  float Kyz = fftKyz[e];
  float Kxz = fftKxz[e];
  float Kxy = fftKxy[e];
  
  fftMx[e  ] = reMx * Kxx + reMy * Kxy + reMz * Kxz;
  fftMx[e+1] = imMx * Kxx + imMy * Kxy + imMz * Kxz;
  
  fftMy[e  ] = reMx * Kxy + reMy * Kyy + reMz * Kyz;
  fftMy[e+1] = imMx * Kxy + imMy * Kyy + imMz * Kyz;
 
  fftMz[e  ] = reMx * Kxz + reMy * Kyz + reMz * Kzz;
  fftMz[e+1] = imMx * Kxz + imMy * Kyz + imMz * Kzz;

}


void gpu_kernel_mul_complex_inplace_symm(float* fftMx,  float* fftMy,  float* fftMz, 
                                         float* fftKxx, float* fftKyy, float* fftKzz,
                                         float* fftKyz, float* fftKxz, float* fftKxy,
                                         int nRealNumbers){
  
  timer_start("kernel_mul");
  assert(nRealNumbers > 0);
  assert(nRealNumbers % 2 == 0);
  
/*  int threadsPerBlock = 512;
  int blocks = (nRealNumbers/2) / threadsPerBlock;
  gpu_checkconf_int(blocks, threadsPerBlock);*/
  
   int gridSize = -1;
   int blockSize = -1;
   make1dconf(nRealNumbers/2, &gridSize, &blockSize);
   printf("gridsize = %d, blockSize = %d\n", gridSize, blockSize);
   
  _gpu_kernel_mul_complex_inplace_symm<<<gridSize, blockSize>>>(
                                      fftMx,  fftMy,  fftMz, 
                                      fftKxx, fftKyy, fftKzz,
                                      fftKyz, fftKxz, fftKxy);
  cudaThreadSynchronize();
  timer_stop("kernel_mul");
}

//_____________________________________________________________________________________________ convolution

void gpuconv2_exec(gpuconv2* conv, tensor* m, tensor* h){
  
  assertDevice(m->list);
  assertDevice(h->list);

  assert(m->rank == 4);
  assert(h->rank == 4);
  for(int i=0; i<4; i++){
    assert(m->size[i] == conv->m->size[i]);
    assert(h->size[i] == conv->h->size[i]);
  }
  
  ///@todo move to setMH()
  conv->m->list = m->list;                              // m, h, mComp and hComp are recycled tensors. We have to set their data each time.
  conv->h->list = h->list;                              // It would be cleaner to have them here as local variables, but this would
  for(int i=0; i<3; i++){                               // mean re-allocating them each time.
    conv->mComp[i]->list = &(m->list[conv->mComp[i]->len * i]);
    conv->hComp[i]->list = &(h->list[conv->hComp[i]->len * i]);
  }
 
  tensor** mComp = conv->mComp;                         // shorthand notations
  tensor** hComp = conv->hComp;
  tensor* fft1 = conv->fft1;
  tensor** fft1Comp = conv->fft1Comp;
  tensor** fft2Comp = conv->fft2Comp;
  
  //_____________________________________________________________________________________________ actual convolution
  
  gpu_zero_tensor(fft1);              // fft1 will now store the zero-padded magnetization
  gpu_zero_tensor(h);
  
  for(int i=0; i<3; i++){
    gpu_copy_pad(mComp[i], fft1Comp[i]);
  }
  
  cudaThreadSynchronize();  ///@todo many redundant syncs
  
  printf("conv2: Been here1\n");
  for(int i=0; i<3; i++){
    gpuFFT3dPlan_forward(conv->fftplan, fft1Comp[i], fft1Comp[i]);  ///@todo out-of-place
  }

  cudaThreadSynchronize();
  
  printf("conv2: Been here2\n");
  gpu_kernel_mul_complex_inplace_symm(fft1Comp[X]->list, fft1Comp[Y]->list, fft1Comp[Z]->list,
                                      conv->fftKernel[X][X]->list, conv->fftKernel[Y][Y]->list, conv->fftKernel[Z][Z]->list, 
                                      conv->fftKernel[Y][Z]->list, conv->fftKernel[X][Z]->list, conv->fftKernel[X][Y]->list,
                                      fft1Comp[X]->len);

  cudaThreadSynchronize();
  
  printf("conv2: Been here3\n");
  for(int i=0; i<3; i++){
    gpuFFT3dPlan_inverse(conv->fftplan, fft1Comp[i], fft1Comp[i]);  ///@todo out-of-place
  }
  
  cudaThreadSynchronize();
  
  for(int i=0; i<3; i++){
    gpu_copy_unpad(fft1Comp[i], hComp[i]);
  }
  
  cudaThreadSynchronize();
}



//_____________________________________________________________________________________________ load kernel

/**
 * the kernel gets normalized
 */
void gpuconv2_loadkernel5DSymm(gpuconv2* conv, tensor* kernel5D){
  
  int* paddedSize = conv->paddedSize;
  
  assert(kernel5D->rank == 5);
  assert(kernel5D->size[0] == 3);
  assert(kernel5D->size[1] == 3);
  assert(kernel5D->size[2+X] == paddedSize[X]);
  assert(kernel5D->size[2+Y] == paddedSize[Y]);
  assert(kernel5D->size[2+Z] == paddedSize[Z]);

  tensor* fftbuffer = conv->fft1Comp[0];
  tensor* KsdDev = new_gputensor(3, paddedSize); /// temp buffer, @todo: free! @todo: not a nice solution
  int* paddedStorageSize = fftbuffer->size;
  gpuFFT3dPlan* plan = new_gpuFFT3dPlan_padded(paddedSize, paddedSize); ///@todo change to paddedstoragesize when gpufft2 is updated

  float N = gpuFFT3dPlan_normalization(plan);
  
  // when the kernel is symmetric, this implementation does a bit redundant work
  for(int s=0; s<3; s++){
    for(int d=0; d<3; d++){
      tensor* Ksd = tensor_component(tensor_component(kernel5D, s), d);

      for(int i=0; i<Ksd->len; i++)
        Ksd->list[i] /= N;
      tensor_copy_to_gpu(Ksd, KsdDev);
      
      gpu_zero_tensor(conv->fftKernel[s][d]);
      gpu_copy_pad(KsdDev, conv->fftKernel[s][d]);
      gpuFFT3dPlan_forward(plan, conv->fftKernel[s][d], conv->fftKernel[s][d]);
    }
  }
  
}

// void gpuconv2_loadkernel(gpuconv2* conv, tensor* kernel){
//   fprintf(stderr, "loadkernel %d x %d x %d\n", kernel->size[2], kernel->size[3], kernel->size[4]);
//   
//   gpuconv2_checksize_kernel(conv, kernel);
//   gpu_plan3d_real_input* plan = new_gpu_plan3d_real_input(kernel->size[2], kernel->size[3], kernel->size[4], NO_ZERO_PAD);
//   float norm = 1.0/float(conv->fftplan->paddedN);
//   float* complex_kernel_ij = new_ram_array(conv->len_ft_kernel_ij);
//   for(int i=0; i<3; i++){
//       for(int j=0; j<3; j++){
// 	
// 	/// @todo !!!!!!!!!!!!!!!!!!!!!!! 
// 	//memcpy_r2c(tensor_get(kernel, 5, i, j, 0, 0, 0), complex_kernel_ij, conv->len_kernel_ij);
// 	
// 	//normalize
// 	for(int e=0; e<conv->len_ft_kernel_ij; e++){
// 	  complex_kernel_ij[e] *= norm;
// 	}
// 	memcpy_to_gpu(complex_kernel_ij, conv->ft_kernel[i][j], conv->len_ft_kernel_ij);
// 	//extract("kernel_ij", conv->ft_kernel[i][j], conv->paddedComplexSize);
// 	gpu_plan3d_real_input_forward(plan, conv->ft_kernel[i][j]);
// 	//extract("ft_kernel_ij", conv->ft_kernel[i][j], conv->paddedComplexSize);
//     }
//   }
//   free(complex_kernel_ij);
//   //delete_gpu_plan3d_real_input(plan);
// }




//_____________________________________________________________________________________________ new gpuconv2

gpuconv2* new_gpuconv2(int* size, int* kernelSize){
//   for(int i=0; i<3; i++){
//     assert(2*size[i] == kernelSize[i]); // generalize later
//   }
  
  gpuconv2* conv = (gpuconv2*)malloc(sizeof(gpuconv2));
  
  int* size4D = new int[4];
  size4D[0] = 3;
  size4D[1] = size[X];
  size4D[2] = size[Y];
  size4D[3] = size[Z];
  
  conv->paddedSize = kernelSize; ///@todo copy, to be sure (goes for all sizes)
  int* paddedSize = conv->paddedSize;
  
  int* paddedStorageSize = new int[3];  ///@todo obtain from fftplan instead
  paddedStorageSize[X] = paddedSize[X];
  paddedStorageSize[Y] = paddedSize[Y];
  paddedStorageSize[Z] = gpu_pad_to_stride(paddedSize[Z] + 2);
  
  int* paddedStorageSize4D = new int[4];
  paddedStorageSize4D[0] = 3;
  paddedStorageSize4D[1] = paddedStorageSize[X];
  paddedStorageSize4D[2] = paddedStorageSize[Y];
  paddedStorageSize4D[3] = paddedStorageSize[Z];
  

  // initialize the FFT plan
  ///@todo generalize !!
  int* zeroPad = new int[3];
  for(int i=0; i<3; i++){
    zeroPad[i] = 1; // todo !!
  }
  conv->fftplan = new_gpuFFT3dPlan_padded(size, kernelSize);	// it's important to FIRST initialize the fft plan because it stores the sizes used by other functions.
  
  conv->m = as_tensorN(NULL, 4, size4D);  // m->list will be set to whatever data is convolved at a certain time.
  conv->h = as_tensorN(NULL, 4, size4D);  // h->list will be set to whatever convolution destination used at a certain time.
  
  conv->fft1 = new_gputensor(4, paddedStorageSize4D);
  conv->fft2 = conv->fft1;  // in-place by default
  
  for(int i=0; i<3; i++){
    conv->mComp[i] = as_tensorN(NULL, 3, size); // note: as_tensor instead of as_tensorN did not gave compilation error and was very difficult to debug...
    conv->hComp[i] = as_tensorN(NULL, 3, size);
    
    conv->fft1Comp[i] = tensor_component(conv->fft1, i);
    conv->fft2Comp[i] = conv->fft1Comp[i]; // in-place by default
  }
  
   for(int i=0; i<3; i++)
    assert(conv->fftplan->paddedStorageSize[i] == conv->fft1Comp[0]->size[i]);

  // By default, the kernel is assumed to by symmetric. Should this not be the case, then the sub-diagonal elements should be separately allocated.
  conv->fftKernel[X][X] = new_gputensor(3, paddedStorageSize);
  conv->fftKernel[Y][Y] = new_gputensor(3, paddedStorageSize);
  conv->fftKernel[Z][Z] = new_gputensor(3, paddedStorageSize);
  
  conv->fftKernel[Y][Z] = new_gputensor(3, paddedStorageSize);
  conv->fftKernel[X][Z] = new_gputensor(3, paddedStorageSize);
  conv->fftKernel[X][Y] = new_gputensor(3, paddedStorageSize);
  
  ///@todo not storing the sub-diagonal elements does not work: they are somehow not loaded.
  conv->fftKernel[Z][Y] = new_gputensor(3, paddedStorageSize);//conv->fftKernel[Y][Z];
  conv->fftKernel[Z][X] = new_gputensor(3, paddedStorageSize);//conv->fftKernel[X][Z];
  conv->fftKernel[Y][X] = new_gputensor(3, paddedStorageSize);//conv->fftKernel[X][Y];

  ///@todo free some sizes
  return conv;
}



#ifdef __cplusplus
}
#endif