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

__global__ void _gpuconv2_copy_pad(float* source, float* dest, int N0, int N1, int N2){

  int i = blockIdx.x;
  int j = blockIdx.y;
  int k = threadIdx.x;

  dest[i*(2*N1)*(2*N2)* + j*(2*N2) + k] = source[i*N1*N2 + j*N2 + k];
}

void gpuconv2_copy_pad(gpuconv2* conv, float* source, float* dest){
  int N0 = conv->fftplan->size[X];
  int N1 = conv->fftplan->size[Y];
  int N2 = conv->fftplan->size[Z];

  dim3 gridSize(N0, N1, 1); ///@todo generalize!
  dim3 blockSize(N2, 1, 1);
  gpu_checkconf(gridSize, blockSize);

  _gpuconv2_copy_pad<<<gridSize, blockSize>>>(source, dest, N0, N1, N2);
  cudaThreadSynchronize();
}


__global__ void _gpuconv2_copy_unpad(float* source, float* dest, int N0, int N1, int N2){
  int i = blockIdx.x;
  int j = blockIdx.y;
  int k = threadIdx.x;

  dest[i*N1*N2 + j*N2 + k] = source[i*(2*N1)*(2*N2)* + j*(2*N2) + k];
}

void gpuconv2_copy_unpad(gpuconv2* conv, float* source, float* dest){
  int N0 = conv->fftplan->size[X];
  int N1 = conv->fftplan->size[Y];
  int N2 = conv->fftplan->size[Z];

  dim3 gridSize(N0, N1, 1); ///@todo generalize!
  dim3 blockSize(N2, 1, 1);
  gpu_checkconf(gridSize, blockSize);

  _gpuconv2_copy_unpad<<<gridSize, blockSize>>>(source, dest, N0, N1, N2);
  cudaThreadSynchronize();
}

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

//_____________________________________________________________________________________________ new

// void gpuconv2_init_kernel(gpuconv2* conv, tensor* kernel){
//   conv->len_kernel_ij = conv->fftplan->paddedN;		//the length of each kernel component K[i][j] (eg: Kxy)
//   conv->len_ft_kernel_ij = conv->fftplan->paddedStorageN;	//the length of each FFT'ed kernel component ~K[i][j] (eg: ~Kxy)
//   gpuconv2_alloc_ft_kernel(conv);
//   gpuconv2_loadkernel(conv, kernel);
// }
// 
// void gpuconv2_init_m(gpuconv2* conv){
//   conv->len_m = 3 * conv->fftplan->N;
//   conv->len_m_comp = conv->fftplan->N;
//   assert(0);
//   // len_ft_m_i unitialized
//   conv->ft_m_i = new_gpu_array(conv->len_ft_m_i);
// }
// 
// void gpuconv2_init_h(gpuconv2* conv){
//   conv->len_h = conv->len_m;
//   conv->len_h_comp = conv->len_m_comp; 
//   conv->len_ft_h = 3 * conv->len_ft_m_i;
//   conv->ft_h = new_gpu_array(conv->len_ft_h);
//   conv->len_ft_h_comp = conv->len_ft_m_i;
//   conv->ft_h_comp = (float**)calloc(3, sizeof(float*));
//   for(int i=0; i<3; i++){ 
//     conv->ft_h_comp[i] = &(conv->ft_h[i * conv->len_ft_h_comp]); // slice the contiguous ft_h array in 3 equal pieces, one for each component
//   }
// }

//_____________________________________________________________________________________________ new gpuconv2

gpuconv2* new_gpuconv2(int* size, tensor* kernel5D){
  
  gpuconv2* conv = (gpuconv2*)malloc(sizeof(gpuconv2));
  
  int* size4D = new int[4];
  size4D[0] = 3;
  size4D[1] = size[X];
  size4D[2] = size[Y];
  size4D[3] = size[Z];
  
  int* paddedSize = new int[3];
  paddedSize[X] = kernel5D->size[2 + X];  // kernel is 5D: 3 x 3 x Xsize x Ysize x Zsize
  paddedSize[Y] = kernel5D->size[2 + Y];
  paddedSize[Z] = kernel5D->size[2 + Z];
  
  int* paddedStorageSize = new int[3];  ///@todo obtain from fftplan instead
  paddedStorageSize[X] = paddedSize[X];
  paddedStorageSize[Y] = paddedSize[Y];
  paddedStorageSize[Z] = paddedSize[Z] + gpu_stride_float(); // we only need + 2 but we take more to remain aligned in the memory
  
  int* paddedStorageSize4D = new int[4];
  paddedStorageSize4D[0] = 3;
  paddedStorageSize4D[1] = paddedStorageSize[X];
  paddedStorageSize4D[2] = paddedStorageSize[Y];
  paddedStorageSize4D[3] = paddedStorageSize[Z];
  
 
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
    conv->mComp[i] = tensor_component(conv->m, i);
    conv->mComp[i] = tensor_component(conv->h, i);
    conv->fft1Comp[i] = tensor_component(conv->fft1, i);
    conv->fft2Comp[i] = conv->fft1Comp[i]; // in-place by default
  }
  
  
  return conv;
}



#ifdef __cplusplus
}
#endif