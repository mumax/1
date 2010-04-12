#include "gpuconv1.h"
#include "gputil.h"
#include "timer.h"
#include <stdio.h>
#include <assert.h>

#ifdef __cplusplus
extern "C" {
#endif

void gpuconv1_init_m_comp(gpuconv1* sim, float* m){
  for(int i=0; i<3; i++){ 			// slice the contiguous m array in 3 equal pieces, one for each component
    sim->m_comp[i] = &(m[i * sim->len_m_comp]); 
  }
}

void gpuconv1_init_h_comp(gpuconv1* sim, float* h){
  for(int i=0; i<3; i++){ 			// slice the contiguous m array in 3 equal pieces, one for each component
    sim->h_comp[i] = &(h[i * sim->len_h_comp]); 
  }
}

void extract(const char* msg, float* data, int* size){
  int N0 = size[0];
  int N1 = size[1];
  int N2 = size[2];
  
  printf("%s(%d x %d x %d){\n", msg, N0, N1, N2);
  tensor* t = new_tensor(3, N0, N1, N2);
  memcpy_from_gpu(data, t->list, tensor_length(t));
  format_tensor(t, stdout);
  printf("}\n\n");
}

//_____________________________________________________________________________________________ convolution

void gpuconv1_exec(gpuconv1* sim, float* m, float* h){
  gpuconv1_init_m_comp(sim, m);
  gpuconv1_init_h_comp(sim, h);
  gpu_zero(sim->ft_h, sim->len_ft_h);							// zero-out field (h) components
  for(int i=0; i<3; i++){								// transform and convolve per magnetization component m_i
    gpu_zero(sim->ft_m_i, sim->len_ft_m_i);						// zero-out the padded magnetization buffer first
    gpu_copy_pad_r2c(sim->m_comp[i], sim->ft_m_i, sim->size[0], sim->size[1], sim->size[2]);	//copy mi into the padded magnetization buffer, converting to complex format
    //extract("ft_m_i", sim->ft_m_i, sim->paddedComplexSize);
    gpuc2cplan_exec(sim->fftplan, sim->ft_m_i, CUFFT_FORWARD);
    //extract("ft_m_i (transformed)", sim->ft_m_i, sim->paddedComplexSize);
    // TODO: asynchronous execution hazard !!
    for(int j=0; j<3; j++){								// apply kernel multiplication to FFT'ed magnetization and add to FFT'ed H-components
      gpu_kernel_mul(sim->ft_m_i, sim->ft_kernel[i][j], sim->ft_h_comp[j], sim->len_ft_m_i);
      //extract("ft_h_j", sim->ft_h_comp[j], sim->paddedComplexSize);
    }
  }
  
  for(int i=0; i<3; i++){
    gpuc2cplan_exec(sim->fftplan, sim->ft_h_comp[i], CUFFT_INVERSE);		// Inplace backtransform of each of the padded h[i]-buffers
    gpu_copy_unpad_c2r(sim->ft_h_comp[i], sim->h_comp[i], sim->size[0], sim->size[1], sim->size[2]);
  }
}

__global__ void _gpu_kernel_mul(float* ft_m_i, float* ft_kernel_ij, float* ft_h_j){
  int e = 2 * ((blockIdx.x * blockDim.x) + threadIdx.x);
  
  float rea = ft_m_i[e];
  float reb = ft_kernel_ij[e];
  float ima = ft_m_i[e + 1];
  float imb = ft_kernel_ij[e + 1];
  ft_h_j[e] 	+=  rea*reb - ima*imb;
  ft_h_j[e + 1] +=  rea*imb + ima*reb;
    
}


void gpu_kernel_mul(float* ft_m_i, float* ft_kernel_ij, float* ft_h_comp_j, int nRealNumbers){
  timer_start("gpuconv1_kernel_mul");
  assert(nRealNumbers > 0);
  assert(nRealNumbers % 2 == 0);
  int threadsPerBlock = 512;
  int blocks = (nRealNumbers/2) / threadsPerBlock;
  gpu_checkconf_int(blocks, threadsPerBlock);
  _gpu_kernel_mul<<<blocks, threadsPerBlock>>>(ft_m_i, ft_kernel_ij, ft_h_comp_j);
  cudaThreadSynchronize();
  timer_stop("gpuconv1_kernel_mul");
}

//_____________________________________________________________________________________________ padding/complex numbers

__global__ void _gpu_copy_unpad_c2r(float* source, float* dest, int N0, int N1, int N2){
  int i = blockIdx.x;
  int j = blockIdx.y;
  int k = threadIdx.x;
  
  //dest[i*N1*N2 + j*N2 + k] = source[i*2*N1*2*2*N2 + j*2*2*N2 + 2*k];
  dest[(i*N1 + j)*N2 + k] = source[(i*N1*8 + j*4)*N2 + 2*k];
}

void gpu_copy_unpad_c2r(float* source, float* dest, int N0, int N1, int N2){
  timer_start("gpuconv1_copy_unpad_c2r");
  dim3 gridsize(N0, N1, 1);
  dim3 blocksize(N2, 1, 1);
  gpu_checkconf(gridsize, blocksize);
  _gpu_copy_unpad_c2r<<<gridsize, blocksize>>>(source, dest, N0, N1, N2);
  cudaThreadSynchronize();
  timer_stop("gpuconv1_copy_unpad_c2r");
}

// Ni: logical size of m
// Todo: not the slightest bit optimized
// arrays should be strided
// N0, N1, N2 are redundant
// linear -> multidimensional array acces
__global__ void _gpu_copy_pad_r2c(float* source, float* dest, int N0, int N1, int N2){
  int i = blockIdx.x;
  int j = blockIdx.y;
  int k = threadIdx.x;
  
  dest[i*2*N1*2*2*N2 + j*2*2*N2 + 2*k  ] = source[i*N1*N2 + j*N2 + k];
  dest[i*2*N1*2*2*N2 + j*2*2*N2 + 2*k+1] = 0.;
  
}

void gpu_copy_pad_r2c(float* source, float* dest, int N0, int N1, int N2){
  /*
   * CUDA Note: the block size must be 2 dimensional (A x B x 1), despite being of type dim3.
   */
  timer_start("gpuconv1_copy_pad_r2c");
  dim3 gridsize(N0, N1, 1);
  dim3 blocksize(N2, 1, 1);
  gpu_checkconf(gridsize, blocksize);
  _gpu_copy_pad_r2c<<<gridsize, blocksize>>>(source, dest, N0, N1, N2);
  cudaThreadSynchronize();
  timer_stop("gpuconv1_copy_pad_r2c");
}

//_____________________________________________________________________________________________ kernel

void gpuconv1_checksize_kernel(gpuconv1* sim, tensor* kernel){
  // kernel should be rank 5 tensor with size 3 x 3 x 2*N0 x 2xN1 x 2xN2 (could be reduced a bit)
  assert(kernel->rank == 5);
  assert(kernel->size[0] == 3);
  assert(kernel->size[1] == 3);
  for(int i=0; i<3; i++){ assert(kernel->size[i+2] == 2 * sim->size[i]); }
  
  assert(kernel->size[2] == sim->paddedSize[0]);
  assert(kernel->size[3] == sim->paddedSize[1]);
  assert(kernel->size[4] == sim->paddedSize[2]);
}

void gpuconv1_alloc_ft_kernel(gpuconv1* sim){
  sim->ft_kernel = (float***)calloc(3, sizeof(float**));
  for(int i=0; i<3; i++){ 
    sim->ft_kernel[i] = (float**)calloc(3, sizeof(float*));
    for(int j=0; j<3; j++){
      sim->ft_kernel[i][j] = new_gpu_array(sim->len_ft_kernel_ij);
    }
  }
}

void gpuconv1_loadkernel(gpuconv1* sim, tensor* kernel){
  fprintf(stderr, "loadkernel %d x %d x %d\n", kernel->size[2], kernel->size[3], kernel->size[4]);
  
  gpuconv1_checksize_kernel(sim, kernel);
  gpuc2cplan* plan = new_gpuc2cplan(kernel->size[2], kernel->size[3], kernel->size[4]);
  float norm = 1.0/float(sim->paddedN);
  float* complex_kernel_ij = new_ram_array(sim->len_ft_kernel_ij);
  for(int i=0; i<3; i++){
      for(int j=0; j<3; j++){
	memcpy_r2c(tensor_get(kernel, 5, i, j, 0, 0, 0), complex_kernel_ij, sim->len_kernel_ij);
	//normalize
	for(int e=0; e<sim->len_ft_kernel_ij; e++){
	  complex_kernel_ij[e] *= norm;
	}
	memcpy_to_gpu(complex_kernel_ij, sim->ft_kernel[i][j], sim->len_ft_kernel_ij);
	//extract("kernel_ij", sim->ft_kernel[i][j], sim->paddedComplexSize);
	gpuc2cplan_exec(plan, sim->ft_kernel[i][j], CUFFT_FORWARD);
	//extract("ft_kernel_ij", sim->ft_kernel[i][j], sim->paddedComplexSize);
    }
  }
  free(complex_kernel_ij);
  delete_gpuc2cplan(plan);
}

//_____________________________________________________________________________________________ new

void gpuconv1_init_sizes(gpuconv1* sim, int N0, int N1, int N2){
  sim->size = (int*)calloc(3, sizeof(int));
  sim->size[0] = N0; sim->size[1] = N1; sim->size[2] = N2;
  sim->N = N0 * N1 * N2;
  
  sim->paddedSize = (int*)calloc(3, sizeof(int));
  sim->paddedSize[0] = 2 * N0; sim->paddedSize[1] = 2 * N1; sim->paddedSize[2] = 2 * N2;
  sim->paddedN = sim->paddedSize[0] * sim->paddedSize[1] * sim->paddedSize[2];
  
  sim->paddedComplexSize = (int*)calloc(3, sizeof(int));
  sim->paddedComplexSize[0] = 2 * N0; sim->paddedComplexSize[1] = 2 * N1; sim->paddedComplexSize[2] = 2 * 2 * N2;
  sim->paddedComplexN = sim->paddedComplexSize[0] * sim->paddedComplexSize[1] * sim->paddedComplexSize[2];
}

void gpuconv1_init_kernel(gpuconv1* sim, tensor* kernel){
  sim->len_kernel_ij = sim->paddedN;		//the length of each kernel component K[i][j] (eg: Kxy)
  sim->len_ft_kernel_ij = sim->paddedComplexN;	//the length of each FFT'ed kernel component ~K[i][j] (eg: ~Kxy)
  gpuconv1_alloc_ft_kernel(sim);
  gpuconv1_loadkernel(sim, kernel);
}

void gpuconv1_init_m(gpuconv1* sim){
  sim->len_m = 3 * sim->N;
  
  sim->len_m_comp = sim->N; 
  sim->m_comp = (float**)calloc(3, sizeof(float*));
  
  sim->len_ft_m_i = sim->paddedComplexN;
  sim->ft_m_i = new_gpu_array(sim->len_ft_m_i);
}


void gpuconv1_init_h(gpuconv1* sim){
  sim->len_h = sim->len_m;
  //sim->h = new_gpu_array(sim->len_h);
  sim->len_h_comp = sim->len_m_comp; 
  sim->h_comp = (float**)calloc(3, sizeof(float*));
//   for(int i=0; i<3; i++){ 
//     sim->h_comp[i] = &(sim->h[i * sim->len_h_comp]); // slice the contiguous h array in 3 equal pieces, one for each component
//   }
  
  sim->len_ft_h = 3 * sim->len_ft_m_i;
  sim->ft_h = new_gpu_array(sim->len_ft_h);
  
  sim->len_ft_h_comp = sim->len_ft_m_i;
  sim->ft_h_comp = (float**)calloc(3, sizeof(float*));
  for(int i=0; i<3; i++){ 
    sim->ft_h_comp[i] = &(sim->ft_h[i * sim->len_ft_h_comp]); // slice the contiguous ft_h array in 3 equal pieces, one for each component
  }
  
}

gpuconv1* new_gpuconv1(int N0, int N1, int N2, tensor* kernel){
  gpuconv1* sim = (gpuconv1*)malloc(sizeof(gpuconv1));
  gpuconv1_init_sizes(sim, N0, N1, N2);
  gpuconv1_init_kernel(sim, kernel);
  gpuconv1_init_m(sim);
  gpuconv1_init_h(sim);
  sim->fftplan = new_gpuc2cplan(sim->paddedSize[0], sim->paddedSize[1], sim->paddedSize[2]);
  return sim;
}


void memcpy_r2c(float* source, float* dest, int nReal){
  for(int i=0; i<nReal; i++){
    dest[2*i]     = source[i];
    dest[2*i + 1] = 0.;
  }
}

#ifdef __cplusplus
}
#endif