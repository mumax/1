#include "gpuconv2.h"
#include "gputil.h"
#include "timer.h"
#include <stdio.h>
#include <assert.h>

#ifdef __cplusplus
extern "C" {
#endif

void gpuconv2_init_m_comp(gpuconv2* conv, float* m){
//   for(int i=0; i<3; i++){ 			// slice the contiguous m array in 3 equal pieces, one for each component
//     conv->m_comp[i] = &(m[i * conv->len_m_comp]); 
//   }
}

void gpuconv2_init_h_comp(gpuconv2* conv, float* h){
//   for(int i=0; i<3; i++){ 			// slice the contiguous m array in 3 equal pieces, one for each component
//     conv->h_comp[i] = &(h[i * conv->len_h_comp]); 
//   }
}

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

void gpuconv2_exec(gpuconv2* conv, float* m, float* h){
//   gpuconv2_init_m_comp(conv, m);
//   gpuconv2_init_h_comp(conv, h);
//   gpu_zero(conv->ft_h, conv->len_ft_h);							// zero-out field (h) components
//   for(int i=0; i<3; i++){								// transform and convolve per magnetization component m_i
//     gpu_zero(conv->ft_m_i, conv->len_ft_m_i);						// zero-out the padded magnetization buffer first
//     gpu_copy_pad_r2c(conv->m_comp[i], conv->ft_m_i, conv->size[0], conv->size[1], conv->size[2]);	//copy mi into the padded magnetization buffer, converting to complex format
//     //extract("ft_m_i", conv->ft_m_i, conv->paddedComplexSize);
//     gpur2cplan_exec(conv->fftplan, conv->ft_m_i, CUFFT_FORWARD);
//     //extract("ft_m_i (transformed)", conv->ft_m_i, conv->paddedComplexSize);
//     // TODO: asynchronous execution hazard !!
//     for(int j=0; j<3; j++){								// apply kernel multiplication to FFT'ed magnetization and add to FFT'ed H-components
//       gpu_kernel_mul(conv->ft_m_i, conv->ft_kernel[i][j], conv->ft_h_comp[j], conv->len_ft_m_i);
//       //extract("ft_h_j", conv->ft_h_comp[j], conv->paddedComplexSize);
//     }
//   }
//   
//   for(int i=0; i<3; i++){
//     gpur2cplan_exec(conv->fftplan, conv->ft_h_comp[i], CUFFT_INVERSE);		// Inplace backtransform of each of the padded h[i]-buffers
//     gpu_copy_unpad_c2r(conv->ft_h_comp[i], conv->h_comp[i], conv->size[0], conv->size[1], conv->size[2]);
//   }
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
  timer_start("gpuconv2_kernel_mul");
//   assert(nRealNumbers > 0);
//   assert(nRealNumbers % 2 == 0);
//   int threadsPerBlock = 512;
//   int blocks = (nRealNumbers/2) / threadsPerBlock;
//   gpu_checkconf_int(blocks, threadsPerBlock);
//   _gpu_kernel_mul<<<blocks, threadsPerBlock>>>(ft_m_i, ft_kernel_ij, ft_h_comp_j);
//   cudaThreadSynchronize();
  timer_stop("gpuconv2_kernel_mul");
}

//_____________________________________________________________________________________________ kernel

void gpuconv2_checksize_kernel(gpuconv2* conv, tensor* kernel){
  // kernel should be rank 5 tensor with size 3 x 3 x 2*N0 x 2xN1 x 2xN2 (could be reduced a bit)
  /*assert(kernel->rank == 5);
  assert(kernel->size[0] == 3);
  assert(kernel->size[1] == 3);
  for(int i=0; i<3; i++){ assert(kernel->size[i+2] == 2 * conv->size[i]); }
  
  assert(kernel->size[2] == conv->paddedSize[0]);
  assert(kernel->size[3] == conv->paddedSize[1]);
  assert(kernel->size[4] == conv->paddedSize[2]);*/
}

void gpuconv2_alloc_ft_kernel(gpuconv2* conv){
//   conv->ft_kernel = (float***)calloc(3, sizeof(float**));
//   for(int i=0; i<3; i++){ 
//     conv->ft_kernel[i] = (float**)calloc(3, sizeof(float*));
//     for(int j=0; j<3; j++){
//       conv->ft_kernel[i][j] = new_gpu_array(conv->len_ft_kernel_ij);
//     }
//   }
}

void gpuconv2_loadkernel(gpuconv2* conv, tensor* kernel){
  fprintf(stderr, "loadkernel %d x %d x %d\n", kernel->size[2], kernel->size[3], kernel->size[4]);
 /* 
  gpuconv2_checksize_kernel(conv, kernel);
  gpur2cplan* plan = new_gpur2cplan(kernel->size[2], kernel->size[3], kernel->size[4]);
  float norm = 1.0/float(conv->paddedN);
  float* complex_kernel_ij = new_ram_array(conv->len_ft_kernel_ij);
  for(int i=0; i<3; i++){
      for(int j=0; j<3; j++){
	memcpy_r2c(tensor_get(kernel, 5, i, j, 0, 0, 0), complex_kernel_ij, conv->len_kernel_ij);
	//normalize
	for(int e=0; e<conv->len_ft_kernel_ij; e++){
	  complex_kernel_ij[e] *= norm;
	}
	memcpy_to_gpu(complex_kernel_ij, conv->ft_kernel[i][j], conv->len_ft_kernel_ij);
	//extract("kernel_ij", conv->ft_kernel[i][j], conv->paddedComplexSize);
	gpur2cplan_exec(plan, conv->ft_kernel[i][j], CUFFT_FORWARD);
	//extract("ft_kernel_ij", conv->ft_kernel[i][j], conv->paddedComplexSize);
    }
  }
  free(complex_kernel_ij);
  delete_gpur2cplan(plan);*/
}

//_____________________________________________________________________________________________ new

void gpuconv2_init_sizes(gpuconv2* conv, int N0, int N1, int N2){
  /*conv->size = (int*)calloc(3, sizeof(int));
  conv->size[0] = N0; conv->size[1] = N1; conv->size[2] = N2;
  conv->N = N0 * N1 * N2;
  
  conv->paddedSize = (int*)calloc(3, sizeof(int));
  conv->paddedSize[0] = 2 * N0; conv->paddedSize[1] = 2 * N1; conv->paddedSize[2] = 2 * N2;
  conv->paddedN = conv->paddedSize[0] * conv->paddedSize[1] * conv->paddedSize[2];
  
  conv->paddedComplexSize = (int*)calloc(3, sizeof(int));
  conv->paddedComplexSize[0] = 2 * N0; conv->paddedComplexSize[1] = 2 * N1; conv->paddedComplexSize[2] = 2 * 2 * N2;
  conv->paddedComplexN = conv->paddedComplexSize[0] * conv->paddedComplexSize[1] * conv->paddedComplexSize[2];*/
}

void gpuconv2_init_kernel(gpuconv2* conv, tensor* kernel){
  /*conv->len_kernel_ij = conv->paddedN;		//the length of each kernel component K[i][j] (eg: Kxy)
  conv->len_ft_kernel_ij = conv->paddedComplexN;	//the length of each FFT'ed kernel component ~K[i][j] (eg: ~Kxy)
  gpuconv2_alloc_ft_kernel(conv);
  gpuconv2_loadkernel(conv, kernel);*/
}

void gpuconv2_init_m(gpuconv2* conv){
  /*conv->len_m = 3 * conv->N;
  
  conv->len_m_comp = conv->N; 
  conv->m_comp = (float**)calloc(3, sizeof(float*));
  
  conv->len_ft_m_i = conv->paddedComplexN;
  conv->ft_m_i = new_gpu_array(conv->len_ft_m_i);*/
}


void gpuconv2_init_h(gpuconv2* conv){
//   conv->len_h = conv->len_m;
//   //conv->h = new_gpu_array(conv->len_h);
//   conv->len_h_comp = conv->len_m_comp; 
//   conv->h_comp = (float**)calloc(3, sizeof(float*));
//   for(int i=0; i<3; i++){ 
//     conv->h_comp[i] = &(conv->h[i * conv->len_h_comp]); // slice the contiguous h array in 3 equal pieces, one for each component
//   }
  /*
  conv->len_ft_h = 3 * conv->len_ft_m_i;
  conv->ft_h = new_gpu_array(conv->len_ft_h);
  
  conv->len_ft_h_comp = conv->len_ft_m_i;
  conv->ft_h_comp = (float**)calloc(3, sizeof(float*));
  for(int i=0; i<3; i++){ 
    conv->ft_h_comp[i] = &(conv->ft_h[i * conv->len_ft_h_comp]); // slice the contiguous ft_h array in 3 equal pieces, one for each component
  }
  */
}

gpuconv2* new_gpuconv2(int N0, int N1, int N2, tensor* kernel){
//   gpuconv2* conv = (gpuconv2*)malloc(sizeof(gpuconv2));
//   gpuconv2_init_sizes(conv, N0, N1, N2);
//   gpuconv2_init_kernel(conv, kernel);
//   gpuconv2_init_m(conv);
//   gpuconv2_init_h(conv);
//   conv->fftplan = new_gpur2cplan(conv->paddedSize[0], conv->paddedSize[1], conv->paddedSize[2]);
//   return conv;
}


//_____________________________________________________________________________________________ FFT

gpur2cplan* new_gpur2cplan(int N0, int N1, int N2){
//   gpur2cplan* plan = (gpur2cplan*) malloc(sizeof(gpur2cplan));
//   gpu_safe( cufftPlan3d(&(plan->handle), N0, N1, N2, CUFFT_C2C) );
//   return plan;
}

void gpur2cplan_exec(gpur2cplan* plan, float* data){
  timer_start("gpur2cplan_exec");
//   gpu_safe( 
//     cufftExecC2C(plan->handle, (cufftComplex*)data, (cufftComplex*)data, direction) 
//   );
//   cudaThreadSynchronize();
  timer_stop("gpur2cplan_exec");
}

void delete_gpur2cplan(gpur2cplan* plan){
  
}

#ifdef __cplusplus
}
#endif