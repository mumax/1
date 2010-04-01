#include "gpusim.h"
#include <stdio.h>
#include <assert.h>

#ifdef __cplusplus
extern "C" {
#endif

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

//_____________________________________________________________________________________________ timestepping


__global__ void _gpu_eulerstep(float* mx, float* my, float* mz, float* hx, float* hy, float* hz, float dt){
  int i = ((blockIdx.x * blockDim.x) + threadIdx.x);
  float alpha = 0.2;
 
  // - m cross H
  float _mxHx = -my[i] * hz[i] + hy[i] * mz[i];
  float _mxHy =  mx[i] * hz[i] - hx[i] * mz[i];
  float _mxHz = -mx[i] * hy[i] + hx[i] * my[i];

  // - m cross (m cross H)
  float _mxmxHx =  my[i] * _mxHz - _mxHy * mz[i];
  float _mxmxHy = -mx[i] * _mxHz + _mxHx * mz[i];
  float _mxmxHz =  mx[i] * _mxHy - _mxHx * my[i];

  float torquex = (_mxHx + _mxmxHx * alpha);
  float torquey = (_mxHy + _mxmxHy * alpha);
  float torquez = (_mxHz + _mxmxHz * alpha);
  
  mx[i] += torquex * dt;
  my[i] += torquey * dt;
  mz[i] += torquez * dt;
  
  float norm = rsqrtf(mx[i]*mx[i] + my[i]*my[i] + mz[i]*mz[i]); // inverse square root
  mx[i] *= norm;
  my[i] *= norm;
  mz[i] *= norm;
}

void gpusim_eulerstep(gpusim* sim, float dt){
  int threadsPerBlock = 512;
  gpusim_updateh(sim);
  int blocks = (sim->len_m_comp) / threadsPerBlock;
  gpu_checkconf_int(blocks, threadsPerBlock);
  _gpu_eulerstep<<<blocks, threadsPerBlock>>>(sim->m_comp[0], sim->m_comp[1], sim->m_comp[2], sim->h_comp[0], sim->h_comp[1], sim->h_comp[2], dt);
}

//_____________________________________________________________________________________________ convolution

void gpusim_updateh(gpusim* sim){
 
  gpu_zero(sim->ft_h, sim->len_ft_h);							// zero-out field (h) components
  for(int i=0; i<3; i++){								// transform and convolve per magnetization component m_i
    gpu_zero(sim->ft_m_i, sim->len_ft_m_i);						// zero-out the padded magnetization buffer first
    gpu_copy_pad_r2c(sim->m_comp[i], sim->ft_m_i, sim->size[0], sim->size[1], sim->size[2]);	//copy mi into the padded magnetization buffer, converting to complex format
    //extract("ft_m_i", sim->ft_m_i, sim->paddedComplexSize);
    gpusim_c2cplan_exec(sim->fftplan, sim->ft_m_i, CUFFT_FORWARD);
    //extract("ft_m_i (transformed)", sim->ft_m_i, sim->paddedComplexSize);
    // TODO: asynchronous execution hazard !!
    for(int j=0; j<3; j++){								// apply kernel multiplication to FFT'ed magnetization and add to FFT'ed H-components
      gpu_kernel_mul(sim->ft_m_i, sim->ft_kernel[i][j], sim->ft_h_comp[j], sim->len_ft_m_i);
      //extract("ft_h_j", sim->ft_h_comp[j], sim->paddedComplexSize);
    }
  }
  
  for(int i=0; i<3; i++){
    gpusim_c2cplan_exec(sim->fftplan, sim->ft_h_comp[i], CUFFT_INVERSE);		// Inplace backtransform of each of the padded h[i]-buffers
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
  assert(nRealNumbers > 0);
  assert(nRealNumbers % 2 == 0);
  int threadsPerBlock = 512;
  int blocks = (nRealNumbers/2) / threadsPerBlock;
  gpu_checkconf_int(blocks, threadsPerBlock);
  _gpu_kernel_mul<<<blocks, threadsPerBlock>>>(ft_m_i, ft_kernel_ij, ft_h_comp_j);
}

//_____________________________________________________________________________________________ padding/complex numbers

__global__ void _gpu_copy_unpad_c2r(float* source, float* dest, int N0, int N1, int N2){
  int i = blockIdx.x;
  int j = blockIdx.y;
  int k = threadIdx.x;
  
  dest[i*N1*N2 + j*N2 + k] = source[i*2*N1*2*2*N2 + j*2*2*N2 + 2*k];
  // 0. == source[i*2*N1*2*2*N2 + j*2*2*N2 + 2*k+1]; // TODO: check this
}

void gpu_copy_unpad_c2r(float* source, float* dest, int N0, int N1, int N2){
  dim3 gridsize(N0, N1, 1);
  dim3 blocksize(N2, 1, 1);
  gpu_checkconf(gridsize, blocksize);
  _gpu_copy_unpad_c2r<<<gridsize, blocksize>>>(source, dest, N0, N1, N2);
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
  dim3 gridsize(N0, N1, 1);
  dim3 blocksize(N2, 1, 1);
  gpu_checkconf(gridsize, blocksize);
  _gpu_copy_pad_r2c<<<gridsize, blocksize>>>(source, dest, N0, N1, N2);
}

//_____________________________________________________________________________________________ load / store data

void gpusim_checksize_m(gpusim* sim, tensor* m){
   // m should be a rank 4 tensor with size 3 x N0 x N1 x N2
  assert(m->rank == 4);
  assert(m->size[0] == 3); 
  for(int i=0; i<3; i++){ assert(m->size[i+1] == sim->size[i]); }
}

void gpusim_loadm(gpusim* sim, tensor* m){
  gpusim_checksize_m(sim, m); 
  memcpy_to_gpu(m->list, sim->m, sim->len_m);
}

void gpusim_storem(gpusim* sim, tensor* m){
   gpusim_checksize_m(sim, m);
   memcpy_from_gpu(sim->m, m->list, sim->len_m);
}

//_____________________________________________________________________________________________ kernel

void gpusim_checksize_kernel(gpusim* sim, tensor* kernel){
  // kernel should be rank 5 tensor with size 3 x 3 x 2*N0 x 2xN1 x 2xN2 (could be reduced a bit)
  assert(kernel->rank == 5);
  assert(kernel->size[0] == 3);
  assert(kernel->size[1] == 3);
  for(int i=0; i<3; i++){ assert(kernel->size[i+2] == 2 * sim->size[i]); }
  
  assert(kernel->size[2] == sim->paddedSize[0]);
  assert(kernel->size[3] == sim->paddedSize[1]);
  assert(kernel->size[4] == sim->paddedSize[2]);
}

void gpusim_alloc_ft_kernel(gpusim* sim){
  sim->ft_kernel = (float***)calloc(3, sizeof(float**));
  for(int i=0; i<3; i++){ 
    sim->ft_kernel[i] = (float**)calloc(3, sizeof(float*));
    for(int j=0; j<3; j++){
      sim->ft_kernel[i][j] = new_gpu_array(sim->len_ft_kernel_ij);
    }
  }
}

void gpusim_loadkernel(gpusim* sim, tensor* kernel){
  fprintf(stderr, "loadkernel %d x %d x %d\n", kernel->size[2], kernel->size[3], kernel->size[4]);
  
  gpusim_checksize_kernel(sim, kernel);
  gpusim_c2cplan* plan = new_gpusim_c2cplan(kernel->size[2], kernel->size[3], kernel->size[4]);
  //float norm = 1.0/float(sim->paddedN);
  float* complex_kernel_ij = new_ram_array(sim->len_ft_kernel_ij);
  for(int i=0; i<3; i++){
      for(int j=0; j<3; j++){
	memcpy_r2c(tensor_get(kernel, 5, i, j, 0, 0, 0), complex_kernel_ij, sim->len_kernel_ij);
	// normalize
// 	for(int e=0; e<sim->len_ft_kernel_ij; e++){
// 	  complex_kernel_ij[e] *= norm;
// 	}
	memcpy_to_gpu(complex_kernel_ij, sim->ft_kernel[i][j], sim->len_ft_kernel_ij);
	//extract("kernel_ij", sim->ft_kernel[i][j], sim->paddedComplexSize);
	gpusim_c2cplan_exec(plan, sim->ft_kernel[i][j], CUFFT_FORWARD);
	//extract("ft_kernel_ij", sim->ft_kernel[i][j], sim->paddedComplexSize);
    }
  }
  free(complex_kernel_ij);
  delete_gpusim_c2cplan(plan);
}

//_____________________________________________________________________________________________ new

void gpusim_init_sizes(gpusim* sim, int N0, int N1, int N2){
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

void gpusim_init_kernel(gpusim* sim, tensor* kernel){
  sim->len_kernel_ij = sim->paddedN;		//the length of each kernel component K[i][j] (eg: Kxy)
  sim->len_ft_kernel_ij = sim->paddedComplexN;	//the length of each FFT'ed kernel component ~K[i][j] (eg: ~Kxy)
  gpusim_alloc_ft_kernel(sim);
  gpusim_loadkernel(sim, kernel);
}

void gpusim_init_m(gpusim* sim){
  sim->len_m = 3 * sim->N;
  sim->m = new_gpu_array(sim->len_m);
  sim->len_m_comp = sim->N; 
  sim->m_comp = (float**)calloc(3, sizeof(float*));
  for(int i=0; i<3; i++){ 			// slice the contiguous m array in 3 equal pieces, one for each component
    sim->m_comp[i] = &(sim->m[i * sim->len_m_comp]); 
  }
  sim->len_ft_m_i = sim->paddedComplexN;
  sim->ft_m_i = new_gpu_array(sim->len_ft_m_i);
}

void gpusim_init_h(gpusim* sim){
  sim->len_h = sim->len_m;
  sim->h = new_gpu_array(sim->len_h);
  sim->len_h_comp = sim->len_m_comp; 
  sim->h_comp = (float**)calloc(3, sizeof(float*));
  for(int i=0; i<3; i++){ 
    sim->h_comp[i] = &(sim->h[i * sim->len_h_comp]); // slice the contiguous h array in 3 equal pieces, one for each component
  }
  
  sim->len_ft_h = 3 * sim->len_ft_m_i;
  sim->ft_h = new_gpu_array(sim->len_ft_h);
  
  sim->len_ft_h_comp = sim->len_ft_m_i;
  sim->ft_h_comp = (float**)calloc(3, sizeof(float*));
  for(int i=0; i<3; i++){ 
    sim->ft_h_comp[i] = &(sim->ft_h[i * sim->len_ft_h_comp]); // slice the contiguous ft_h array in 3 equal pieces, one for each component
  }
  
}

gpusim* new_gpusim(int N0, int N1, int N2, tensor* kernel){
  gpusim* sim = (gpusim*)malloc(sizeof(gpusim));
  gpusim_init_sizes(sim, N0, N1, N2);
  gpusim_init_kernel(sim, kernel);
  gpusim_init_m(sim);
  gpusim_init_h(sim);
  sim->fftplan = new_gpusim_c2cplan(sim->paddedSize[0], sim->paddedSize[1], sim->paddedSize[2]);
  return sim;
}

//_____________________________________________________________________________________________ FFT

gpusim_c2cplan* new_gpusim_c2cplan(int N0, int N1, int N2){
  gpusim_c2cplan* plan = (gpusim_c2cplan*) malloc(sizeof(gpusim_c2cplan));
  gpusim_safe( cufftPlan3d(&(plan->handle), N0, N1, N2, CUFFT_C2C) );
  return plan;
}

void gpusim_c2cplan_exec(gpusim_c2cplan* plan, float* data, int direction){
  gpusim_safe( 
    cufftExecC2C(plan->handle, (cufftComplex*)data, (cufftComplex*)data, direction) 
  );
}

void delete_gpusim_c2cplan(gpusim_c2cplan* plan){
  //gpusim_safe( cudaFree(plan->gpudata) );
  // TODO: free handle
  free(plan);
}

//_____________________________________________________________________________________________ data management


int gpu_len(int size){
  assert(size > 0);
  int threadsPerBlock = 512; // todo: centralize
  int gpulen = ((size-1)/threadsPerBlock + 1) * threadsPerBlock;
  assert(gpulen % threadsPerBlock == 0);
  assert(gpulen > 0);
  return gpulen;
}

__global__ void _gpu_zero(float* list){
  int i = (blockIdx.x * blockDim.x) + threadIdx.x;
  list[i] = 0.;
}

// can probably be replaced by some cudaMemset function,
// but I just wanted to try out some manual cuda coding.
void gpu_zero(float* data, int nElements){
  int threadsPerBlock = 512;
  assert(nElements > 0);
  int blocks = nElements / threadsPerBlock;
  gpu_checkconf_int(blocks, threadsPerBlock);
  _gpu_zero<<<blocks, threadsPerBlock>>>(data);
}

void memcpy_to_gpu(float* source, float* dest, int nElements){
  assert(nElements > 0);
  int status = cudaMemcpy(dest, source, nElements*sizeof(float), cudaMemcpyHostToDevice);
  if(status != cudaSuccess){
    fprintf(stderr, "CUDA could not copy %d floats from host addres %p to device addres %p\n", nElements, source, dest);
    gpusim_safe(status);
  }
}


void memcpy_from_gpu(float* source, float* dest, int nElements){
  assert(nElements > 0);
  int status = cudaMemcpy(dest, source, nElements*sizeof(float), cudaMemcpyDeviceToHost);
  if(status != cudaSuccess){
    fprintf(stderr, "CUDA could not copy %d floats from device addres %p to host addres %p\n", nElements, source, dest);
    gpusim_safe(status);
  }
}

// does not seem to work.. 
void memcpy_gpu_to_gpu(float* source, float* dest, int nElements){
  assert(nElements > 0);
  int status = cudaMemcpy(dest, source, nElements*sizeof(float), cudaMemcpyHostToHost);
  if(status != cudaSuccess){
    fprintf(stderr, "CUDA could not copy %d floats from host addres %p to host addres %p\n", nElements, source, dest);
    gpusim_safe(status);
  }
}

// todo: we need cudaMalloc3D for better alignment!
float* new_gpu_array(int size){
  assert(size > 0);
  int threadsPerBlock = 512;
  assert(size % threadsPerBlock == 0);
  float* array = NULL;
  int status = cudaMalloc((void**)(&array), size * sizeof(float));
  if(status != cudaSuccess){
    fprintf(stderr, "CUDA could not allocate %d floats\n", size);
    gpusim_safe(status);
  }
  //assert(array != NULL); // strange: it seems cuda can return 0 as a valid address?? 
  if(array == 0){
    fprintf(stderr, "cudaMalloc(%p, %ld) returned null without error status, retrying...\n", (void**)(&array), size * sizeof(float));
    abort();
  }
  return array;
}

float* new_ram_array(int size){
  assert(size > 0);
  float* array = (float*)calloc(size, sizeof(float));
  if(array == NULL){
    fprintf(stderr, "could not allocate %d floats in main memory\n", size);
    abort();
  }
  return array;
}

void memcpy_r2c(float* source, float* dest, int nReal){
  for(int i=0; i<nReal; i++){
    dest[2*i]     = source[i];
    dest[2*i + 1] = 0.;
  }
}

//_____________________________________________________________________________________________ misc

#define MAX_THREADS_PER_BLOCK 512
#define MAX_BLOCKSIZE_X 512
#define MAX_BLOCKSIZE_Y 512
#define MAX_BLOCKSIZE_Z 64

#define MAX_GRIDSIZE_Z 1

void gpu_checkconf(dim3 gridsize, dim3 blocksize){
  assert(blocksize.x * blocksize.y * blocksize.z <= MAX_THREADS_PER_BLOCK);
  assert(blocksize.x * blocksize.y * blocksize.z >  0);
  
  assert(blocksize.x <= MAX_BLOCKSIZE_X);
  assert(blocksize.y <= MAX_BLOCKSIZE_Y);
  assert(blocksize.z <= MAX_BLOCKSIZE_Z);
  
  assert(gridsize.z <= MAX_GRIDSIZE_Z);
  assert(gridsize.x > 0);
  assert(gridsize.y > 0);
  assert(gridsize.z > 0);
}

void gpu_checkconf_int(int gridsize, int blocksize){
  assert(blocksize <= MAX_BLOCKSIZE_X);
  assert(gridsize > 0);
}

void gpusim_safe(int status){
  if(status != cudaSuccess){
    fprintf(stderr, "received CUDA error: %s\n", cudaGetErrorString((cudaError_t)status));
    abort();
  }
}

#ifdef __cplusplus
}
#endif