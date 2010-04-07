#include "gputil.h"
#include <stdio.h>
#include <assert.h>

#ifdef __cplusplus
extern "C" {
#endif


gpuc2cplan* new_gpuc2cplan(int N0, int N1, int N2){
  gpuc2cplan* plan = (gpuc2cplan*) malloc(sizeof(gpuc2cplan));
  gpu_safe( cufftPlan3d(&(plan->handle), N0, N1, N2, CUFFT_C2C) );
  return plan;
}

void gpuc2cplan_exec(gpuc2cplan* plan, float* data, int direction){
  gpu_safe( 
    cufftExecC2C(plan->handle, (cufftComplex*)data, (cufftComplex*)data, direction) 
  );
}

void delete_gpuc2cplan(gpuc2cplan* plan){
  //gpu_safe( cudaFree(plan->gpudata) );
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

float* new_gpu_array(int size){
  assert(size > 0);
  int threadsPerBlock = 512;
  assert(size % threadsPerBlock == 0);
  float* array = NULL;
  int status = cudaMalloc((void**)(&array), size * sizeof(float));
  if(status != cudaSuccess){
    fprintf(stderr, "CUDA could not allocate %d floats\n", size);
    gpu_safe(status);
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
  cudaThreadSynchronize();
}

void memcpy_to_gpu(float* source, float* dest, int nElements){
  assert(nElements > 0);
  int status = cudaMemcpy(dest, source, nElements*sizeof(float), cudaMemcpyHostToDevice);
  if(status != cudaSuccess){
    fprintf(stderr, "CUDA could not copy %d floats from host addres %p to device addres %p\n", nElements, source, dest);
    gpu_safe(status);
  }
}


void memcpy_from_gpu(float* source, float* dest, int nElements){
  assert(nElements > 0);
  int status = cudaMemcpy(dest, source, nElements*sizeof(float), cudaMemcpyDeviceToHost);
  if(status != cudaSuccess){
    fprintf(stderr, "CUDA could not copy %d floats from device addres %p to host addres %p\n", nElements, source, dest);
    gpu_safe(status);
  }
}

// does not seem to work.. 
void memcpy_gpu_to_gpu(float* source, float* dest, int nElements){
  assert(nElements > 0);
  int status = cudaMemcpy(dest, source, nElements*sizeof(float), cudaMemcpyDeviceToDevice);
  if(status != cudaSuccess){
    fprintf(stderr, "CUDA could not copy %d floats from device addres %p to device addres %p\n", nElements, source, dest);
    gpu_safe(status);
  }
  cudaThreadSynchronize();
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

void gpu_safe(int status){
  if(status != cudaSuccess){
    fprintf(stderr, "received CUDA error: %s\n", cudaGetErrorString((cudaError_t)status));
    abort();
  }
}

#ifdef __cplusplus
}
#endif