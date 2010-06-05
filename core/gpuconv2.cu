#include "gpufft.h"
#include "gpuconv2.h"
#include "gputil.h"
#include "timer.h"
#include <stdio.h>
#include <assert.h>

#ifdef __cplusplus
extern "C" {
#endif

/** For debugging: gets tensor from the GPU and prints to screen */
// void extract(const char* msg, float* data, int* size){
//   int N0 = size[0];
//   int N1 = size[1];
//   int N2 = size[2];
//   
//   printf("%s(%d x %d x %d){\n", msg, N0, N1, N2);
//   tensor* t = new_tensor(3, N0, N1, N2);
//   memcpy_from_gpu(data, t->list, tensor_length(t));
//   format_tensor(t, stdout);
//   printf("}\n\n");
// }

//_____________________________________________________________________________________________ convolution

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
   // dest must not be larger than source
  for(int i=0; i<3; i++){
    assert(source->size[i] >= dest->size[i]);
  }
  
  int D0 = source->size[X];
  int D1 = source->size[Y];
  int D2 = source->size[Z];

  dim3 gridSize(D0, D1, 1); ///@todo generalize!
  dim3 blockSize(D2, 1, 1);
  gpu_checkconf(gridSize, blockSize);

  _gpuconv2_copy_pad<<<gridSize, blockSize>>>(source->list, dest->list, source->size[1], source->size[2], D1, D2);
  cudaThreadSynchronize();
}



// void gpuconv2_copy_pad(gpuconv2* conv, float* source, float* dest){
//   int N0 = conv->fftplan->size[X];
//   int N1 = conv->fftplan->size[Y];
//   int N2 = conv->fftplan->size[Z];
// 
//   dim3 gridSize(N0, N1, 1); ///@todo generalize!
//   dim3 blockSize(N2, 1, 1);
//   gpu_checkconf(gridSize, blockSize);
// 
//   _gpuconv2_copy_pad<<<gridSize, blockSize>>>(source, dest, N0, N1, N2);
//   cudaThreadSynchronize();
// }
// 
// 
// __global__ void _gpuconv2_copy_unpad(float* source, float* dest, int N0, int N1, int N2){
//   int i = blockIdx.x;
//   int j = blockIdx.y;
//   int k = threadIdx.x;
// 
//   dest[i*N1*N2 + j*N2 + k] = source[i*(2*N1)*(2*N2)* + j*(2*N2) + k];
// }
// 
// void gpuconv2_copy_unpad(gpuconv2* conv, float* source, float* dest){
//   int N0 = conv->fftplan->size[X];
//   int N1 = conv->fftplan->size[Y];
//   int N2 = conv->fftplan->size[Z];
// 
//   dim3 gridSize(N0, N1, 1); ///@todo generalize!
//   dim3 blockSize(N2, 1, 1);
//   gpu_checkconf(gridSize, blockSize);
// 
//   _gpuconv2_copy_unpad<<<gridSize, blockSize>>>(source, dest, N0, N1, N2);
//   cudaThreadSynchronize();
// }

void gpuconv2_exec(gpuconv2* conv, float* m, float* h){
//   // set m and h components
//   float* m_comp[3];
//   float* h_comp[3];
//   for(int i=0; i<3; i++){ 			// slice the contiguous m array in 3 equal pieces, one for each component
//     m_comp[i] = &(m[i * conv->len_m_comp]); 
//     h_comp[i] = &(h[i * conv->len_h_comp]); 
//   }
  
  /*
  gpu_zero(conv->ft_h, conv->len_ft_h);							// zero-out field (h) components
  for(int i=0; i<3; i++){								// transform and convolve per magnetization component m_i
    gpu_zero(conv->ft_m_i, conv->len_ft_m_i);						// zero-out the padded magnetization buffer first
    gpu_copy_pad_r2c(conv->m_comp[i], conv->ft_m_i, conv->size[0], conv->size[1], conv->size[2]);	//copy mi into the padded magnetization buffer, converting to complex format
    //extract("ft_m_i", conv->ft_m_i, conv->paddedComplexSize);
		
		
    gpu_plan3d_real_input_exec(conv->fftplan, conv->ft_m_i, CUFFT_FORWARD);
    //extract("ft_m_i (transformed)", conv->ft_m_i, conv->paddedComplexSize);
    // TODO: asynchronous execution hazard !!
		
		
    for(int j=0; j<3; j++){								// apply kernel multiplication to FFT'ed magnetization and add to FFT'ed H-components
      gpu_kernel_mul(conv->ft_m_i, conv->ft_kernel[i][j], conv->ft_h_comp[j], conv->len_ft_m_i);
      //extract("ft_h_j", conv->ft_h_comp[j], conv->paddedComplexSize);
    }
  }
//  TODO: Save memory by performing gpu_kernel_mul + FFT_inverse + copy_unpad for each component in serie, memory savings are 2*paddedStorageN
  
  for(int i=0; i<3; i++){
    gpu_plan3d_real_input_exec(conv->fftplan, conv->ft_h_comp[i], CUFFT_INVERSE);		// Inplace backtransform of each of the padded h[i]-buffers
    gpu_copy_unpad_c2r(conv->ft_h_comp[i], conv->h_comp[i], conv->size[0], conv->size[1], conv->size[2]);
  }*/
}

__global__ void _gpu_kernel_mul2(float* ft_m_i, float* ft_kernel_ij, float* ft_h_j){
  /*int e = 2 * ((blockIdx.x * blockDim.x) + threadIdx.x);
  
  float rea = ft_m_i[e];
  float reb = ft_kernel_ij[e];
  float ima = ft_m_i[e + 1];
  float imb = ft_kernel_ij[e + 1];
  ft_h_j[e] 	+=  rea*reb - ima*imb;
  ft_h_j[e + 1] +=  rea*imb + ima*reb;
  */  
}


void gpu_kernel_mul2(float* ft_m_i, float* ft_kernel_ij, float* ft_h_comp_j, int nRealNumbers){
/*  timer_start("gpuconv2_kernel_mul");
//   assert(nRealNumbers > 0);
//   assert(nRealNumbers % 2 == 0);
//   int threadsPerBlock = 512;
//   int blocks = (nRealNumbers/2) / threadsPerBlock;
//   gpu_checkconf_int(blocks, threadsPerBlock);
//   _gpu_kernel_mul<<<blocks, threadsPerBlock>>>(ft_m_i, ft_kernel_ij, ft_h_comp_j);
//   cudaThreadSynchronize();
  timer_stop("gpuconv2_kernel_mul");*/
}

//_____________________________________________________________________________________________ kernel

// void gpuconv2_checksize_kernel(gpuconv2* conv, tensor* kernel){
//   // kernel should be rank 5 tensor with size 3 x 3 x 2*N0 x 2xN1 x 2xN2 (could be reduced a bit)
// //   assert(kernel->rank == 5);
// //   assert(kernel->size[0] == 3);
// //   assert(kernel->size[1] == 3);
// //   for(int i=0; i<3; i++){ assert(kernel->size[i+2] == 2 * conv->size[i]); }
// //   
// //   assert(kernel->size[2] == conv->paddedSize[0]);
// //   assert(kernel->size[3] == conv->paddedSize[1]);
// //   assert(kernel->size[4] == conv->paddedSize[2]);
// }
// 
// void gpuconv2_alloc_ft_kernel(gpuconv2* conv){
//   conv->ft_kernel = (float***)calloc(3, sizeof(float**));
//   for(int i=0; i<3; i++){ 
//     conv->ft_kernel[i] = (float**)calloc(3, sizeof(float*));
//     for(int j=0; j<3; j++){
//       conv->ft_kernel[i][j] = new_gpu_array(conv->len_ft_kernel_ij);
//     }
//   }
// }

int* NO_ZERO_PAD = (int*)calloc(3, sizeof(int));

void gpuconv2_loadkernel5DSymm(gpuconv2* conv, tensor* kernel5D){
  int* paddedSize = conv->paddedSize;
  
  assert(kernel5D->rank == 5);
  assert(kernel5D->size[0] == 3);
  assert(kernel5D->size[1] == 3);
  assert(kernel5D->size[2+X] == paddedSize[X]);
  assert(kernel5D->size[2+Y] == paddedSize[Y]);
  assert(kernel5D->size[2+Z] == paddedSize[Z]);

  gpu_plan3d_real_input* plan = new_gpu_plan3d_real_input(paddedSize[X], paddedSize[Y], paddedSize[Z], NO_ZERO_PAD);

  
  
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
  paddedStorageSize[Z] = paddedSize[Z] + gpu_stride_float(); // we only need + 2 but we take more to remain aligned in the memory
  
  int* paddedStorageSize4D = new int[4];
  paddedStorageSize4D[0] = 3;
  paddedStorageSize4D[1] = paddedStorageSize[X];
  paddedStorageSize4D[2] = paddedStorageSize[Y];
  paddedStorageSize4D[3] = paddedStorageSize[Z];
  

  // initialize the FFT plan
  ///@todo generalize !!
  int* zeroPad = new int[3];
  for(int i=0; i<3; i++){
    zeroPad[i] = 1;
  }
  conv->fftplan = new_gpu_plan3d_real_input(size[X], size[Y], size[Z], zeroPad);	// it's important to FIRST initialize the fft plan because it stores the sizes used by other functions.
  
  
  conv->m = as_tensorN(NULL, 4, size4D);  // m->list will be set to whatever data is convolved at a certain time.
  conv->h = as_tensor(NULL, 4, size4D);  // h->list will be set to whatever convolution destination used at a certain time.
  
  conv->fft1 = new_gputensor(4, paddedStorageSize4D);
  conv->fft2 = conv->fft1;  // in-place by default
  
  for(int i=0; i<3; i++){
    conv->mComp[i] = as_tensor(NULL, 3, size);
    conv->hComp[i] = as_tensor(NULL, 3, size);
    
    conv->fft1Comp[i] = tensor_component(conv->fft1, i);
    conv->fft2Comp[i] = conv->fft1Comp[i]; // in-place by default
  }

  // By default, the kernel is assumed to by symmetric. Should this not be the case, then the sub-diagonal elements should be separately allocated.
  conv->fftKernel[X][X] = new_gputensor(3, paddedStorageSize);
  conv->fftKernel[Y][Y] = new_gputensor(3, paddedStorageSize);
  conv->fftKernel[Z][Z] = new_gputensor(3, paddedStorageSize);
  
  conv->fftKernel[Y][Z] = new_gputensor(3, paddedStorageSize);
  conv->fftKernel[X][Z] = new_gputensor(3, paddedStorageSize);
  conv->fftKernel[X][Y] = new_gputensor(3, paddedStorageSize);
  
  conv->fftKernel[Z][Y] = conv->fftKernel[Y][Z];
  conv->fftKernel[Z][X] = conv->fftKernel[X][Z];
  conv->fftKernel[Y][X] = conv->fftKernel[X][Y];


  ///@todo free some sizes
  return conv;
}



#ifdef __cplusplus
}
#endif