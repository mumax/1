#include "gputil.h"
#include "timer.h"
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <stdarg.h>

#ifdef __cplusplus
extern "C" {
#endif

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
  if(size % threadsPerBlock != 0){
    fprintf(stderr, "WARNING: new_gpu_array: size %% threadsPerBlock != 0\n");
  }
  float* array = NULL;
  int status = cudaMalloc((void**)(&array), size * sizeof(float));
  if(status != cudaSuccess){
    fprintf(stderr, "CUDA could not allocate %d floats\n", size);
    gpu_safe(status);
  }
  assert(array != NULL); // strange: it seems cuda can return 0 as a valid address?? 
  if(array == 0){
     fprintf(stderr, "cudaMalloc(%p, %d) returned null without error status, retrying...\n", (void**)(&array), (int)(size * sizeof(float))); // (int) cast to resolve 32-bit vs. 64 bit problem...
    abort();
  }
  return array;
}

// tensor* new_gpu_tensor(int rank, ...){
//   
//   int* size = (int*)safe_calloc(rank, sizeof(int));
//   int N = 1;
//   
//   va_list varargs;
//   va_start(varargs, rank);
//   
//   for(int i=0; i<rank; i++){
//     size[i] = va_arg(varargs, int);
//     N *= size[i];
//   }
//   va_end(varargs);
//   
//   return as_tensorN(new_gpu_array(N), rank, size);
// }


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
  timer_start("gpu_zero");
  
  int threadsPerBlock = 512;
  assert(nElements > 0);
  int blocks = nElements / threadsPerBlock;
  
  gpu_checkconf_int(blocks, threadsPerBlock);
  _gpu_zero<<<blocks, threadsPerBlock>>>(data);
  cudaThreadSynchronize();
  
  timer_stop("gpu_zero");
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
  timer_start("memcpy_gpu_to_gpu");
  
  assert(nElements > 0);
  int status = cudaMemcpy(dest, source, nElements*sizeof(float), cudaMemcpyDeviceToDevice);
  if(status != cudaSuccess){
    fprintf(stderr, "CUDA could not copy %d floats from device addres %p to device addres %p\n", nElements, source, dest);
    gpu_safe(status);
  }
  cudaThreadSynchronize();
  
  timer_stop("memcpy_gpu_to_gpu");
}



//_____________________________________________________________________________________________ misc

/* We test for the optimal array stride by creating a 1x1 matrix and checking
 * the stride returned by CUDA.
 */
int gpu_stride_float(){
  size_t width = 1;
  size_t height = 1;
  
  float* devPtr;
  size_t pitch;
  gpu_safe( cudaMallocPitch((void**)&devPtr, &pitch, width * sizeof(float), height) );
  gpu_safe( cudaFree(devPtr) );
  return pitch / sizeof(float);
}

/** @todo use cudaDeviceProperties */
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

void make3dconf(int N0, int N1, int N2, dim3* gridSize, dim3* blockSize ){

}


// struct cudaDeviceProp {
// char name[256];
// size_t totalGlobalMem;
// size_t sharedMemPerBlock;
// int regsPerBlock;
// int warpSize;
// size_t memPitch;
// int maxThreadsPerBlock;
// int maxThreadsDim[3];
// int maxGridSize[3];
// size_t totalConstMem;
// int major;
// int minor;
// int clockRate;
// size_t textureAlignment;
// int deviceOverlap;
// int multiProcessorCount;
// int kernelExecTimeoutEnabled;
// int integrated;
// int canMapHostMemory;
// int computeMode;
// int concurrentKernels;
// }


void print_device_properties(FILE* out){
  int device = -1;
  gpu_safe( cudaGetDevice(&device) );
  
  cudaDeviceProp prop;
  gpu_safe( cudaGetDeviceProperties(&prop, device) ); 
  
  int MiB = 1024 * 1024;
  int kiB = 1024;
  
  fprintf(out, "    Device number: %d\n", device);
  fprintf(out, "      Device name: %s\n", prop.name);
  fprintf(out, "    Global Memory: %d MiB\n", (int)(prop.totalGlobalMem/MiB));
  fprintf(out, "    Shared Memory: %d kiB/block\n", (int)(prop.sharedMemPerBlock/kiB));
  fprintf(out, "        Registers: %d per block\n", (int)(prop.regsPerBlock/kiB));
  fprintf(out, "        Warp size: %d threads\n", (int)(prop.warpSize));
  fprintf(out, " Max memory pitch: %d bytes\n", (int)(prop.memPitch));
  fprintf(out, "Max threads/block: %d\n", prop.maxThreadsPerBlock);
  fprintf(out, "  Constant memory: %d kiB\n", (int)(prop.totalConstMem/kiB));
  
  
  
  
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