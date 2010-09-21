// The code in this source file is based on the reduction code from the CUDPP library. Hence the following notice:

/*
Copyright (c) 2007-2010 The Regents of the University of California, Davis
campus ("The Regents") and NVIDIA Corporation ("NVIDIA"). All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice,
      this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright notice,
      this list of conditions and the following disclaimer in the documentation
      and/or other materials provided with the distribution.
    * Neither the name of the The Regents, nor NVIDIA, nor the names of its
      contributors may be used to endorse or promote products derived from this
      software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

// This code has been significantly modified from its original version.

#include "gpu_reduction.h"
#include "gpu_conf.h"
#include "gpu_safe.h"
#include "gpu_mem.h"

extern "C"
bool isPow2(unsigned int x){
  return ((x&(x-1))==0);
}

// Utility class used to avoid linker errors with extern
// unsized shared memory arrays with templated type
template<class T>
struct SharedMemory {
  __device__ inline operator       T*()
  {
    extern __shared__ int __smem[];
    return (T*)__smem;
  }

  __device__ inline operator const T*() const
  {
    extern __shared__ int __smem[];
    return (T*)__smem;
  }
};

/*
This version adds multiple elements per thread sequentially.  This reduces the overall
cost of the algorithm while keeping the work complexity O(n) and the step complexity O(log n).
(Brent's Theorem optimization)
Note, this kernel needs a minimum of 64*sizeof(T) bytes of shared memory.
In other words if blockSize <= 32, allocate 64*sizeof(T) bytes.
If blockSize > 32, allocate blockSize*sizeof(T) bytes.
*/

/// This kernel takes a partial sum
template <unsigned int blockSize, bool nIsPow2>
__global__ void _gpu_sum_kernel(float* g_idata, float* g_odata, unsigned int n) {
  float* sdata = SharedMemory<float>();

  // perform first level of reduction,
  // reading from global memory, writing to shared memory
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x*blockSize*2 + threadIdx.x;
  unsigned int gridSize = blockSize*2*gridDim.x;

  float mySum = 0;

  // we reduce multiple elements per thread.  The number is determined by the
  // number of active thread blocks (via gridDim).  More blocks will result
  // in a larger gridSize and therefore fewer elements per thread
  while (i < n)
  {
    mySum += g_idata[i];
    // ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
    if (nIsPow2 || i + blockSize < n)
      mySum += g_idata[i+blockSize];
    i += gridSize;
  }

  // each thread puts its local sum into shared memory
  sdata[tid] = mySum;
  __syncthreads();


  // do reduction in shared mem
//   if (blockSize >= 512) { if (tid < 256) { sdata[tid] = mySum = mySum + sdata[tid + 256]; } __syncthreads(); }
//   if (blockSize >= 256) { if (tid < 128) { sdata[tid] = mySum = mySum + sdata[tid + 128]; } __syncthreads(); }
//   if (blockSize >= 128) { if (tid <  64) { sdata[tid] = mySum = mySum + sdata[tid +  64]; } __syncthreads(); }
  if (blockSize >= 512) { if (tid < 256) { mySum = mySum + sdata[tid + 256]; sdata[tid] = mySum; } __syncthreads(); }
  if (blockSize >= 256) { if (tid < 128) { mySum = mySum + sdata[tid + 128]; sdata[tid] = mySum; } __syncthreads(); }
  if (blockSize >= 128) { if (tid <  64) { mySum = mySum + sdata[tid +  64]; sdata[tid] = mySum; } __syncthreads(); }

  if (tid < 32)
    {
      // now that we are using warp-synchronous programming (below)
      // we need to declare our shared memory volatile so that the compiler
      // doesn't reorder stores to it and induce incorrect behavior.
      volatile float* smem = sdata;
      if (blockSize >=  64) { mySum = mySum + smem[tid + 32]; smem[tid] = mySum;  }
      if (blockSize >=  32) { mySum = mySum + smem[tid + 16]; smem[tid] = mySum;  }
      if (blockSize >=  16) { mySum = mySum + smem[tid +  8]; smem[tid] = mySum;  }
      if (blockSize >=   8) { mySum = mySum + smem[tid +  4]; smem[tid] = mySum;  }
      if (blockSize >=   4) { mySum = mySum + smem[tid +  2]; smem[tid] = mySum;  }
      if (blockSize >=   2) { mySum = mySum + smem[tid +  1]; smem[tid] = mySum;  }
    }
    // write result for this block to global mem
    if (tid == 0)
      g_odata[blockIdx.x] = sdata[0];
}

/// This kernel takes a partial maximum
template <unsigned int blockSize, bool nIsPow2>
__global__ void _gpu_max_kernel(float* g_idata, float* g_odata, unsigned int n) {
  float* sdata = SharedMemory<float>();

  // perform first level of reduction,
  // reading from global memory, writing to shared memory
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x*blockSize*2 + threadIdx.x;
  unsigned int gridSize = blockSize*2*gridDim.x;

  float myMax = -1E37;

  // we reduce multiple elements per thread.  The number is determined by the
  // number of active thread blocks (via gridDim).  More blocks will result
  // in a larger gridSize and therefore fewer elements per thread
  while (i < n)
  {
    myMax = fmax(myMax, g_idata[i]);
    // ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
    if (nIsPow2 || i + blockSize < n)
      myMax = fmax(myMax, g_idata[i+blockSize]);
    i += gridSize;
  }

  // each thread puts its local sum into shared memory
  sdata[tid] = myMax;
  __syncthreads();


  // do reduction in shared mem
  if (blockSize >= 512) { if (tid < 256) { myMax = fmax(myMax, sdata[tid + 256]); sdata[tid] = myMax; } __syncthreads(); }
  if (blockSize >= 256) { if (tid < 128) { myMax = fmax(myMax, sdata[tid + 128]); sdata[tid] = myMax; } __syncthreads(); }
  if (blockSize >= 128) { if (tid <  64) { myMax = fmax(myMax, sdata[tid +  64]); sdata[tid] = myMax; } __syncthreads(); }

  if (tid < 32)
    {
      // now that we are using warp-synchronous programming (below)
      // we need to declare our shared memory volatile so that the compiler
      // doesn't reorder stores to it and induce incorrect behavior.
      volatile float* smem = sdata;
      if (blockSize >=  64) { myMax = fmax(myMax, smem[tid + 32]); smem[tid] = myMax;  }
      if (blockSize >=  32) { myMax = fmax(myMax, smem[tid + 16]); smem[tid] = myMax;  }
      if (blockSize >=  16) { myMax = fmax(myMax, smem[tid +  8]); smem[tid] = myMax;  }
      if (blockSize >=   8) { myMax = fmax(myMax, smem[tid +  4]); smem[tid] = myMax;  }
      if (blockSize >=   4) { myMax = fmax(myMax, smem[tid +  2]); smem[tid] = myMax;  }
      if (blockSize >=   2) { myMax = fmax(myMax, smem[tid +  1]); smem[tid] = myMax;  }
    }
    // write result for this block to global mem
    if (tid == 0)
      g_odata[blockIdx.x] = sdata[0];
}

/// This kernel takes a partial maximum of absolute values
template <unsigned int blockSize, bool nIsPow2>
__global__ void _gpu_maxabs_kernel(float* g_idata, float* g_odata, unsigned int n) {
  float* sdata = SharedMemory<float>();

  // perform first level of reduction,
  // reading from global memory, writing to shared memory
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x*blockSize*2 + threadIdx.x;
  unsigned int gridSize = blockSize*2*gridDim.x;

  float myMaxabs = 0.;

  // we reduce multiple elements per thread.  The number is determined by the
  // number of active thread blocks (via gridDim).  More blocks will result
  // in a larger gridSize and therefore fewer elements per thread
  while (i < n)
  {
    myMaxabs = fmax(myMaxabs, fabs(g_idata[i]));
    // ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
    if (nIsPow2 || i + blockSize < n)
      myMaxabs = fmax(myMaxabs, fabs(g_idata[i+blockSize]));
    i += gridSize;
  }

  // each thread puts its local sum into shared memory
  sdata[tid] = myMaxabs;
  __syncthreads();


  // do reduction in shared mem
  if (blockSize >= 512) { if (tid < 256) { myMaxabs = fmax(myMaxabs, sdata[tid + 256]); sdata[tid] = myMaxabs; } __syncthreads(); }
  if (blockSize >= 256) { if (tid < 128) { myMaxabs = fmax(myMaxabs, sdata[tid + 128]); sdata[tid] = myMaxabs; } __syncthreads(); }
  if (blockSize >= 128) { if (tid <  64) { myMaxabs = fmax(myMaxabs, sdata[tid +  64]); sdata[tid] = myMaxabs; } __syncthreads(); }

  if (tid < 32)
    {
      // now that we are using warp-synchronous programming (below)
      // we need to declare our shared memory volatile so that the compiler
      // doesn't reorder stores to it and induce incorrect behavior.
      volatile float* smem = sdata;
      if (blockSize >=  64) { myMaxabs = fmax(myMaxabs, smem[tid + 32]); smem[tid] = myMaxabs;  }
      if (blockSize >=  32) { myMaxabs = fmax(myMaxabs, smem[tid + 16]); smem[tid] = myMaxabs;  }
      if (blockSize >=  16) { myMaxabs = fmax(myMaxabs, smem[tid +  8]); smem[tid] = myMaxabs;  }
      if (blockSize >=   8) { myMaxabs = fmax(myMaxabs, smem[tid +  4]); smem[tid] = myMaxabs;  }
      if (blockSize >=   4) { myMaxabs = fmax(myMaxabs, smem[tid +  2]); smem[tid] = myMaxabs;  }
      if (blockSize >=   2) { myMaxabs = fmax(myMaxabs, smem[tid +  1]); smem[tid] = myMaxabs;  }
    }
    // write result for this block to global mem
    if (tid == 0)
      g_odata[blockIdx.x] = sdata[0];
}

#ifdef __cplusplus
extern "C" {
#endif

void gpu_partial_sums(float* d_idata, float* d_odata, int blocks, int threads, int size) {
  dim3 dimBlock(threads, 1, 1);
  dim3 dimGrid(blocks, 1, 1);

  // when there is only one warp per block, we need to allocate two warps
  // worth of shared memory so that we don't index shared memory out of bounds
  int smemSize = (threads <= 32) ? 2 * threads * sizeof(float) : threads * sizeof(float);

  if (isPow2(size))
  {
    switch (threads)
    {
      case 512: _gpu_sum_kernel<512, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
      case 256: _gpu_sum_kernel<256, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
      case 128: _gpu_sum_kernel<128, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
      case  64: _gpu_sum_kernel< 64, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
      case  32: _gpu_sum_kernel< 32, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
      case  16: _gpu_sum_kernel< 16, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
      case   8: _gpu_sum_kernel<  8, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
      case   4: _gpu_sum_kernel<  4, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
      case   2: _gpu_sum_kernel<  2, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
      case   1: _gpu_sum_kernel<  1, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
    }
  }
  else
  {
    switch (threads)
    {
      case 512: _gpu_sum_kernel<512, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
      case 256: _gpu_sum_kernel<256, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
      case 128: _gpu_sum_kernel<128, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
      case  64: _gpu_sum_kernel< 64, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
      case  32: _gpu_sum_kernel< 32, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
      case  16: _gpu_sum_kernel< 16, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
      case   8: _gpu_sum_kernel<  8, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
      case   4: _gpu_sum_kernel<  4, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
      case   2: _gpu_sum_kernel<  2, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
      case   1: _gpu_sum_kernel<  1, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
    }
  }
  gpu_sync();
}

void gpu_partial_max(float* d_idata, float* d_odata, int blocks, int threads, int size) {
  dim3 dimBlock(threads, 1, 1);
  dim3 dimGrid(blocks, 1, 1);

  // when there is only one warp per block, we need to allocate two warps
  // worth of shared memory so that we don't index shared memory out of bounds
  int smemSize = (threads <= 32) ? 2 * threads * sizeof(float) : threads * sizeof(float);

  if (isPow2(size))
  {
    switch (threads)
    {
      case 512: _gpu_max_kernel<512, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
      case 256: _gpu_max_kernel<256, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
      case 128: _gpu_max_kernel<128, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
      case  64: _gpu_max_kernel< 64, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
      case  32: _gpu_max_kernel< 32, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
      case  16: _gpu_max_kernel< 16, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
      case   8: _gpu_max_kernel<  8, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
      case   4: _gpu_max_kernel<  4, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
      case   2: _gpu_max_kernel<  2, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
      case   1: _gpu_max_kernel<  1, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
    }
  }
  else
  {
    switch (threads)
    {
      case 512: _gpu_max_kernel<512, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
      case 256: _gpu_max_kernel<256, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
      case 128: _gpu_max_kernel<128, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
      case  64: _gpu_max_kernel< 64, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
      case  32: _gpu_max_kernel< 32, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
      case  16: _gpu_max_kernel< 16, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
      case   8: _gpu_max_kernel<  8, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
      case   4: _gpu_max_kernel<  4, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
      case   2: _gpu_max_kernel<  2, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
      case   1: _gpu_max_kernel<  1, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
    }
  }
  gpu_sync();
}

void gpu_partial_maxabs(float* d_idata, float* d_odata, int blocks, int threads, int size) {
  dim3 dimBlock(threads, 1, 1);
  dim3 dimGrid(blocks, 1, 1);

  // when there is only one warp per block, we need to allocate two warps
  // worth of shared memory so that we don't index shared memory out of bounds
  int smemSize = (threads <= 32) ? 2 * threads * sizeof(float) : threads * sizeof(float);

  if (isPow2(size))
  {
    switch (threads)
    {
      case 512: _gpu_maxabs_kernel<512, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
      case 256: _gpu_maxabs_kernel<256, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
      case 128: _gpu_maxabs_kernel<128, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
      case  64: _gpu_maxabs_kernel< 64, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
      case  32: _gpu_maxabs_kernel< 32, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
      case  16: _gpu_maxabs_kernel< 16, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
      case   8: _gpu_maxabs_kernel<  8, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
      case   4: _gpu_maxabs_kernel<  4, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
      case   2: _gpu_maxabs_kernel<  2, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
      case   1: _gpu_maxabs_kernel<  1, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
    }
  }
  else
  {
    switch (threads)
    {
      case 512: _gpu_maxabs_kernel<512, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
      case 256: _gpu_maxabs_kernel<256, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
      case 128: _gpu_maxabs_kernel<128, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
      case  64: _gpu_maxabs_kernel< 64, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
      case  32: _gpu_maxabs_kernel< 32, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
      case  16: _gpu_maxabs_kernel< 16, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
      case   8: _gpu_maxabs_kernel<  8, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
      case   4: _gpu_maxabs_kernel<  4, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
      case   2: _gpu_maxabs_kernel<  2, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
      case   1: _gpu_maxabs_kernel<  1, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
    }
  }
  gpu_sync();
}

float gpu_reduce(int operation, float* input, float* dev2, float* host2, int blocks, int threads, int N){
  switch(operation){
    default: abort(); break;
    case REDUCE_ADD:
    {
      gpu_partial_sums(input, dev2, blocks, threads, N);
      memcpy_from_gpu(dev2, host2, blocks);
      float sum = 0.;
      for(int i=0; i<blocks; i++){
        sum += host2[i];
      }
      return sum;
    }
    case REDUCE_MAX:
    {
      gpu_partial_max(input, dev2, blocks, threads, N);
      memcpy_from_gpu(dev2, host2, blocks);
      float max = 0.;
      for(int i=0; i<blocks; i++){
        max = fmax(max, host2[i]);
      }
      return max;
    }
    case REDUCE_MAXABS:
    {
      gpu_partial_maxabs(input, dev2, blocks, threads, N);
      memcpy_from_gpu(dev2, host2, blocks);
      float maxabs = 0.;
      for(int i=0; i<blocks; i++){
        maxabs = fmax(maxabs, host2[i]);
      }
      return maxabs;
    }

  }
}

///@internal for debugging only, use gpu_reduce() instead of this function
///@todo leaks memory, should not allocate
float gpu_sum(float* data, int N){
  
  int threads = 128;
  while (N <= threads){
    threads /= 2;
  }
  int blocks = divUp(N, threads*2);

  float* dev2 = new_gpu_array(blocks);
  float* host2 = (float*)calloc(blocks, sizeof(float));

  return gpu_reduce(REDUCE_ADD, data, dev2, host2, blocks, threads, N);
}

#ifdef __cplusplus
}
#endif
