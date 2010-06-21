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
//   int threadsPerBlock = 512;
//   if(size % threadsPerBlock != 0){
//     fprintf(stderr, "WARNING: new_gpu_array: size %% threadsPerBlock != 0\n");
//   }
  float* array = NULL;
  gpu_safe( cudaMalloc((void**)(&array), size * sizeof(float)) );
//   if(status != cudaSuccess){
//     fprintf(stderr, "CUDA could not allocate %d floats\n", size);
//     gpu_safe(status);
//   }
  assert(array != NULL); // strange: it seems cuda can return 0 as a valid address?? 
  gpu_zero(array, size);
//   if(array == 0){
//      fprintf(stderr, "cudaMalloc(%p, %d) returned null without error status, retrying...\n", (void**)(&array), (int)(size * sizeof(float))); // (int) cast to resolve 32-bit vs. 64 bit problem...
//     abort();
//   }
  return array;
}

tensor* new_gputensor(int rank, int* size){
  int len = 1;
  for(int i=0; i<rank; i++){
    len *= size[i];
  }
  return as_tensorN(new_gpu_array(len), rank, size);
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

//_____________________________________________________________________________________________ util

void gpu_zero(float* data, int nElements){
  timer_start("gpu_zero");

  gpu_safe( cudaMemset(data, 0, nElements*sizeof(float)) );
  
  timer_stop("gpu_zero");
}


void gpu_zero_tensor(tensor* t){
  gpu_zero(t->list, t->len);
}

void format_gputensor(tensor* t, FILE* out){
  tensor* host = new_tensorN(t->rank, t->size);
  tensor_copy_from_gpu(t, host);
  format_tensor(host, out);
  delete_tensor(host);
}



float* _host_array = NULL;
float* _device_array = NULL;

void assertHost(float* pointer){
  if(_host_array == NULL){
    _host_array = new_ram_array(1);
  }
  _host_array[0] = pointer[0]; // may throw segfault
}

void assertDevice(float* pointer){
  if(_device_array == NULL){
    _device_array = new_gpu_array(1);
  }
  memcpy_gpu_to_gpu(pointer, _device_array, 1); // may throw segfault
}

//_____________________________________________________________________________________________ copy

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

void tensor_copy_to_gpu(tensor* source, tensor* dest){
  assert(tensor_equalsize(source, dest));
  memcpy_to_gpu(source->list, dest->list, source->len);
}

void tensor_copy_from_gpu(tensor* source, tensor* dest){
  assert(tensor_equalsize(source, dest));
  memcpy_from_gpu(source->list, dest->list, source->len);
}

void tensor_copy_gpu_to_gpu(tensor* source, tensor* dest){
  assert(tensor_equalsize(source, dest));
  memcpy_gpu_to_gpu(source->list, dest->list, source->len);
}

//_____________________________________________________________________________________________ misc

// to avoid having to calculate gpu_stide_float over and over,
// we cache the result of the first invocation and return it
// for all subsequent calls.
// (the function itself is rather expensive)
int _gpu_stride_float_cache = -1;

/* We test for the optimal array stride by creating a 1x1 matrix and checking
 * the stride returned by CUDA.
 */
int gpu_stride_float(){
  if( _gpu_stride_float_cache == -1){
    size_t width = 1;
    size_t height = 1;
    
    float* devPtr;
    size_t pitch;
    gpu_safe( cudaMallocPitch((void**)&devPtr, &pitch, width * sizeof(float), height) );
    gpu_safe( cudaFree(devPtr) );
    _gpu_stride_float_cache = pitch / sizeof(float);
    fprintf(stderr, "GPU stride: %d floats\n", _gpu_stride_float_cache);
  }
  return _gpu_stride_float_cache;
}


void gpu_override_stride(int nFloats){
  assert(nFloats > -2);
  fprintf(stderr, "GPU stride overridden to %d floats\n", nFloats);
  _gpu_stride_float_cache = nFloats;
}

int gpu_pad_to_stride(int nFloats){
  assert(nFloats > 0);
  int stride = gpu_stride_float();
  int gpulen = ((nFloats-1)/stride + 1) * stride;
  
  assert(gpulen % stride == 0);
  assert(gpulen > 0);
  assert(gpulen >= nFloats);
  return gpulen;
}

//_____________________________________________________________________________________________ check conf

/** @todo use cudaDeviceProperties */
#define MAX_THREADS_PER_BLOCK 512
#define MAX_BLOCKSIZE_X 512
#define MAX_BLOCKSIZE_Y 512
#define MAX_BLOCKSIZE_Z 64

#define MAX_GRIDSIZE_Z 1

/** @todo: uses deviced props.  */
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

void check3dconf(dim3 gridSize, dim3 blockSize){
 
  cudaDeviceProp* prop = (cudaDeviceProp*)gpu_getproperties();
  int maxThreadsPerBlock = prop->maxThreadsPerBlock;
  int* maxBlockSize = prop->maxThreadsDim;
  int* maxGridSize = prop->maxGridSize;
  
  assert(gridSize.x > 0);
  assert(gridSize.y > 0);
  assert(gridSize.z > 0);
  
  assert(blockSize.x > 0);
  assert(blockSize.y > 0);
  assert(blockSize.z > 0);
  
  assert(blockSize.x <= maxBlockSize[X]);
  assert(blockSize.y <= maxBlockSize[Y]);
  assert(blockSize.z <= maxBlockSize[Z]);
  
  assert(gridSize.x <= maxGridSize[X]);
  assert(gridSize.y <= maxGridSize[Y]);
  assert(gridSize.z <= maxGridSize[Z]);
  
  assert(blockSize.x * blockSize.y * blockSize.z <= maxThreadsPerBlock);
}

void check1dconf(int gridsize, int blocksize){
  assert(gridsize > 0);
  assert(blocksize > 0);
  assert(blocksize <= ((cudaDeviceProp*)gpu_getproperties())->maxThreadsPerBlock);
}

//_____________________________________________________________________________________________ make conf

void _make1dconf(int N, unsigned int* gridSize, unsigned int* blockSize, int maxGridSize, int maxBlockSize){
  if(N >= maxBlockSize){
    *blockSize = maxBlockSize;
    while(N % (*blockSize) != 0){
      (*blockSize)--;
    }
    *gridSize = N / *blockSize;
  }
  else{ // N < maxBlockSize
    *blockSize = N;
    *gridSize = 1;
  }
  check1dconf(*gridSize, *blockSize);
  assert((*blockSize) * (*gridSize) == N);
  assert(*blockSize <= maxBlockSize);
  assert(*gridSize <= maxGridSize);
}

void make1dconf(int N, int* gridSize, int* blockSize){
  cudaDeviceProp* prop = (cudaDeviceProp*)gpu_getproperties();
  _make1dconf(N, (unsigned int*)gridSize, (unsigned int*)blockSize, prop->maxGridSize[X], prop->maxThreadsPerBlock);
}

void make3dconf(int N0, int N1, int N2, dim3* gridSize, dim3* blockSize ){
  
  cudaDeviceProp* prop = (cudaDeviceProp*)gpu_getproperties();
  int maxThreadsPerBlock = prop->maxThreadsPerBlock;
  int* maxBlockSize = prop->maxThreadsDim;
  int* maxGridSize = prop->maxGridSize;
  
  _make1dconf(N2, &(gridSize->z), &(blockSize->z), maxGridSize[Z], maxBlockSize[Z]);
  
  int newMaxBlockSizeY = min(maxBlockSize[Y], maxThreadsPerBlock / blockSize->z);
  _make1dconf(N1, &(gridSize->y), &(blockSize->y), maxGridSize[Y], newMaxBlockSizeY);
  
  int newMaxBlockSizeX = min(maxBlockSize[X], maxThreadsPerBlock / (blockSize->z * blockSize->y));
  _make1dconf(N0, &(gridSize->x), &(blockSize->x), maxGridSize[X], newMaxBlockSizeX);
  
  assert(blockSize->x * gridSize->x == N0);
  assert(blockSize->y * gridSize->y == N1);
  assert(blockSize->z * gridSize->z == N2);
  
  
  fprintf(stderr, "make3dconf(%d, %d, %d): (%d x %d x %d) x (%d x %d x %d)\n", 
	  N0, N1, N2, 
	  gridSize->x, gridSize->y, gridSize->z,
	  blockSize->x, blockSize->y, blockSize->z);
	  
   check3dconf(*gridSize, *blockSize);
}





//_____________________________________________________________________________________________ device properties

cudaDeviceProp* gpu_device_properties = NULL;

void* gpu_getproperties(void){
  if(gpu_device_properties == NULL){
    int device = -1;
    gpu_safe( cudaGetDevice(&device) );
  
    gpu_device_properties = (cudaDeviceProp*)malloc(sizeof(cudaDeviceProp));
    gpu_safe( cudaGetDeviceProperties(gpu_device_properties, device) ); 
  }
  return gpu_device_properties;
}


void print_device_properties(FILE* out){
  int device = -1;
  gpu_safe( cudaGetDevice(&device) );
  
  cudaDeviceProp prop;
  gpu_safe( cudaGetDeviceProperties(&prop, device) ); 
  
  int MiB = 1024 * 1024;
  int kiB = 1024;
  
  fprintf(out, "     Device number: %d\n", device);
  fprintf(out, "       Device name: %s\n", prop.name);
  fprintf(out, "     Global Memory: %d MiB\n", (int)(prop.totalGlobalMem/MiB));
  fprintf(out, "     Shared Memory: %d kiB/block\n", (int)(prop.sharedMemPerBlock/kiB));
  fprintf(out, "   Constant memory: %d kiB\n", (int)(prop.totalConstMem/kiB));
  fprintf(out, "         Registers: %d per block\n", (int)(prop.regsPerBlock/kiB));
  fprintf(out, "         Warp size: %d threads\n", (int)(prop.warpSize));
  //fprintf(out, "  Max memory pitch: %d bytes\n", (int)(prop.memPitch));
  fprintf(out, " Texture alignment: %d bytes\n", (int)(prop.textureAlignment));
  fprintf(out, " Max threads/block: %d\n", prop.maxThreadsPerBlock);
  fprintf(out, "    Max block size: %d x %d x %d threads\n", prop.maxThreadsDim[X], prop.maxThreadsDim[Y], prop.maxThreadsDim[Z]);
  fprintf(out, "     Max grid size: %d x %d x %d blocks\n", prop.maxGridSize[X], prop.maxGridSize[Y], prop.maxGridSize[Z]);
  fprintf(out, "Compute capability: %d.%d\n", prop.major, prop.minor);
  fprintf(out, "        Clock rate: %d MHz\n", prop.clockRate/1000);
  fprintf(out, "   Multiprocessors: %d\n", prop.multiProcessorCount);
  fprintf(out, "   Timeout enabled: %d\n", prop.kernelExecTimeoutEnabled);
  fprintf(out, "      Compute mode: %d\n", prop.computeMode);
  fprintf(out, "    Device overlap: %d\n", prop.deviceOverlap);
  fprintf(out, "Concurrent kernels: %d\n", prop.concurrentKernels);
  fprintf(out, "        Integrated: %d\n", prop.integrated);
  fprintf(out, "  Can map host mem: %d\n", prop.canMapHostMemory);
  
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