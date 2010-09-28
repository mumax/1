#include "gpu_mem.h"
#include "gpu_safe.h"
#include "gpu_conf.h"
#include "../macros.h"

#ifdef __cplusplus
extern "C" {
#endif

unsigned long long int _gpu_usedmem = 0;


unsigned long long int gpu_usedmem(){
  return _gpu_usedmem;
}


float* new_gpu_array(int size){
  assert(size > 0);
  float* array = NULL;
  gpu_safe( cudaMalloc((void**)(&array), size * sizeof(float)) );
  _gpu_usedmem += size * sizeof(float);
  assert(array != NULL); // strange: it seems cuda can return 0 as a valid address?? 
  gpu_zero(array, size);
  return array;
}

void free_gpu_array(float* ptr){
  gpu_safe(cudaFree(ptr));
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


///@internal kernel
__global__ void _gpu_zero(float* a){
  int i = ((blockIdx.x * blockDim.x) + threadIdx.x);
  a[i] = 0.0f;
}

void gpu_zero(float* data, int nElements){
  debugvv(fprintf(stderr, "gpu_zero(%p, %d)\n", data, nElements));
  gpu_safe( cudaMemset(data, 0, nElements*sizeof(float)) );
  gpu_sync();
//   int gridSize = -1, blockSize = -1;
//   make1dconf(nElements, &gridSize, &blockSize);
//   _gpu_zero<<<gridSize, blockSize>>>(data);
//   gpu_sync();
  
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
  memcpy_on_gpu(pointer, _device_array, 1); // may throw segfault
}

void memcpy_gpu_dir(float* source, float* dest, int nElements, int direction){
  if(direction == 1){
      memcpy_to_gpu(source, dest,nElements);
  }
  else if(direction == 2){
      memcpy_on_gpu(source, dest, nElements);
  }
  else if(direction == 3){
      memcpy_from_gpu(source,  dest, nElements);
  }
  else{
    abort();
  }
}


void memcpy_to_gpu(float* source, float* dest, int nElements){
  assert(nElements > 0);
  int status = cudaMemcpy(dest, source, nElements*sizeof(float), cudaMemcpyHostToDevice);
  if(status != cudaSuccess){
    fprintf(stderr, "CUDA could not copy %d floats from host addres %p to device addres %p\n", nElements, source, dest);
    gpu_safe(status);
  }
  gpu_sync();
}


void memcpy_from_gpu(float* source, float* dest, int nElements){
  debugvv(fprintf(stderr, "memcpy_from_gpu(%p, %p, %d)\n", source, dest, nElements))
  assert(nElements > 0);
  int status = cudaMemcpy(dest, source, nElements*sizeof(float), cudaMemcpyDeviceToHost);
  if(status != cudaSuccess){
    fprintf(stderr, "CUDA could not copy %d floats from device addres %p to host addres %p\n", nElements, source, dest);
    gpu_safe(status);
  }
  gpu_sync(); ///@todo memcpy is synchronous, remove this sync.
}

void memcpy_on_gpu(float* source, float* dest, int nElements){
  assert(nElements > 0);
  int status = cudaMemcpy(dest, source, nElements*sizeof(float), cudaMemcpyDeviceToDevice);
  if(status != cudaSuccess){
    fprintf(stderr, "CUDA could not copy %d floats from device addres %p to device addres %p\n", nElements, source, dest);
    gpu_safe(status);
  }
  gpu_sync();
}

float gpu_array_get(float* dataptr, int index){
  float result = 666.0;
  memcpy_from_gpu(&(dataptr[index]), &result, 1);
  return result;
}


void gpu_array_set(float* dataptr, int index, float value){
  memcpy_to_gpu(&value, &(dataptr[index]), 1);
}


// to avoid having to calculate gpu_stide_float over and over,
// we cache the result of the first invocation and return it
// for all subsequent calls.
// (the function itself is rather expensive)
// -1 means not set yet.

// Arne: FFT strategy:
// It seems a good thing to forget about the gpu stride:
//  * Data sizes are usually quite large powers of 2, so the alignment is already fine.
//  * 1/3 of the FFT's will act on improperly aligned data, but for computation-limited kernels on 2.x GPU's, this should be barely notable
//  * 1/2 of the transposes will be slowed down by about a factor of 2, but R2C FFT's always break the alignment, so there is no workaround anyway.

int _gpu_stride_float_cache = 1;

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
    debugv( fprintf(stderr, "GPU stride: %d floats\n", _gpu_stride_float_cache) );
  }
  return _gpu_stride_float_cache;
}


void gpu_override_stride(int nFloats){
  assert(nFloats > -2);
  debugv( fprintf(stderr, "GPU stride overridden to %d floats\n", nFloats) );
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


#ifdef __cplusplus
}
#endif
