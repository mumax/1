// The code in this source file is based on the reduction code from the CUDPP library. Hence the following notice:

/*
CUDA Data-Parallel Primitives Library (CUDPP) is the proprietary property of The Regents of the University of California ("The Regents") and NVIDIA Corporation ("NVIDIA").
Copyright (c) 2007 The Regents of the University of California, Davis campus and NVIDIA Corporation. All Rights Reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

- Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
- Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
- Neither the name of The Regents, NVIDIA, nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

The end-user understands that the program was developed for research purposes and is advised not to rely exclusively on the program for any reason.

THE SOFTWARE PROVIDED IS ON AN "AS IS" BASIS, AND THE REGENTS, NVIDIA AND CONTRIBUTORS HAVE NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS. THE REGENTS, NVIDIA AND CONTRIBUTORS SPECIFICALLY DISCLAIM ANY EXPRESS OR IMPLIED WARRANTIES,
INCLUDING BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE REGENTS, NVIDIA OR CONTRIBUTORS BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, EXEMPLARY OR
CONSEQUENTIAL DAMAGES, INCLUDING BUT NOT LIMITED TO PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES, LOSS OF USE, DATA OR PROFITS, OR BUSINESS INTERRUPTION, HOWEVER CAUSED AND UNDER ANY THEORY OF LIABILITY WHETHER IN CONTRACT, STRICT LIABILITY OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

If you do not agree to these terms, do not download or use the software.  This license may be modified only in a writing signed by authorized signatory of all parties.  For The Regents contact copyright@ucdavis.edu.

Relating to funding received by the Regents- Acknowledgment: This material is based upon work supported by the Department of Energy under Award Numbers DE-FG02-04ER25609 and DE-FC02-06ER25777.

Disclaimer: This report was prepared as an account of work sponsored by an agency of the United States Government. Neither the United States Government nor any agency thereof, nor any of their employees,
makes any warranty, express or implied, or assumes any legal liability or responsibility for the accuracy, completeness, or usefulness of any information, apparatus, product, or process disclosed, or represents that its use would not infringe privately owned rights. Reference
herein to any specific commercial product, process, or service by trade name, trademark, manufacturer, or otherwise does not necessarily
constitute or imply its endorsement, recommendation, or favoring by the United States Government or any agency thereof. The views and opinions of authors expressed herein do not necessarily state or reflect those of the United States Government or any agency hereof.
*/

// This code has been significantly modified from its original version.

#include "gpu_reduction.h"
#include "gpu_conf.h"
#include "gpu_mem.h"

extern "C"
bool isPow2(unsigned int x)
{
  return ((x&(x-1))==0);
}

// Utility class used to avoid linker errors with extern
// unsized shared memory arrays with templated type
template<class T>
struct SharedMemory
{
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
template <unsigned int blockSize, bool nIsPow2>
__global__ void
reduce6(float* g_idata, float* g_odata, unsigned int n)
{
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
  if (blockSize >= 512) { if (tid < 256) { sdata[tid] = mySum = mySum + sdata[tid + 256]; } __syncthreads(); }
  if (blockSize >= 256) { if (tid < 128) { sdata[tid] = mySum = mySum + sdata[tid + 128]; } __syncthreads(); }
  if (blockSize >= 128) { if (tid <  64) { sdata[tid] = mySum = mySum + sdata[tid +  64]; } __syncthreads(); }


  {
    // now that we are using warp-synchronous programming (below)
    // we need to declare our shared memory volatile so that the compiler
    // doesn't reorder stores to it and induce incorrect behavior.
    volatile float* smem = sdata;
    if (blockSize >=  64) { smem[tid] = mySum = mySum + smem[tid + 32];  }
    if (blockSize >=  32) { smem[tid] = mySum = mySum + smem[tid + 16];  }
    if (blockSize >=  16) { smem[tid] = mySum = mySum + smem[tid +  8];  }
    if (blockSize >=   8) { smem[tid] = mySum = mySum + smem[tid +  4];  }
    if (blockSize >=   4) { smem[tid] = mySum = mySum + smem[tid +  2];  }
    if (blockSize >=   2) { smem[tid] = mySum = mySum + smem[tid +  1];  }
  }

  // write result for this block to global mem
  if (tid == 0)
    g_odata[blockIdx.x] = sdata[0];
}

#ifdef __cplusplus
extern "C" {
  #endif
void _gpu_reduce(int size, int threads, int blocks, float* d_idata, float* d_odata) {
    dim3 dimBlock(threads, 1, 1);
    dim3 dimGrid(blocks, 1, 1);

    // when there is only one warp per block, we need to allocate two warps
    // worth of shared memory so that we don't index shared memory out of bounds
    int smemSize = (threads <= 32) ? 2 * threads * sizeof(float) : threads * sizeof(float);

    if (isPow2(size))
    {
      switch (threads)
      {
        case 512:
          reduce6<512, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
        case 256:
          reduce6<256, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
        case 128:
          reduce6<128, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
        case 64:
          reduce6< 64, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
        case 32:
          reduce6< 32, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
        case 16:
          reduce6< 16, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
        case  8:
          reduce6<  8, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
        case  4:
          reduce6<  4, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
        case  2:
          reduce6<  2, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
        case  1:
          reduce6<  1, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
      }
    }
    else
    {
      switch (threads)
      {
        case 512:
          reduce6<512, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
        case 256:
          reduce6<256, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
        case 128:
          reduce6<128, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
        case 64:
          reduce6< 64, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
        case 32:
          reduce6< 32, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
        case 16:
          reduce6< 16, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
        case  8:
          reduce6<  8, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
        case  4:
          reduce6<  4, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
        case  2:
          reduce6<  2, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
        case  1:
          reduce6<  1, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
      }
    }
  }

///@ todo leaks memory, should not allocate
float gpu_reduce(float* data, int N){

  int threads = 128;
  while (N <= threads){
    threads /= 2;
  }
  int blocks = divUp(N, threads*2);

  float* dev2 = new_gpu_array(blocks);
  float* host2 = new float[blocks];

  _gpu_reduce(N, threads, blocks, data, dev2);

  memcpy_from_gpu(dev2, host2, blocks);

  float sum = 0.;

  for(int i=0; i<blocks; i++){
    sum += host2[i];
  }
  //gpu_free(dev2);
  //delete[] host2;
  return sum;
}

#ifdef __cplusplus
}
#endif
