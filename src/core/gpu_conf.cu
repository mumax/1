#include "gpu_conf.h"
#include "gpu_properties.h"
#include "../macros.h"
#include <assert.h>
#include "gputil.h"

#ifdef __cplusplus
extern "C" {
#endif

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

  ///// HACK ////
  if(maxBlockSize > 128){
    debugvv( fprintf(stderr, "WARNING: using 128 as max block size! \n") );
    maxBlockSize = 128;
  }

  
  //if(N >= maxBlockSize){
    *blockSize = maxBlockSize;
    while(N % (*blockSize) != 0){
      (*blockSize)/=2;
    }
    *gridSize = N / *blockSize;
//   }
//   else{ // N < maxBlockSize
//     *blockSize = N;
//     *gridSize = 1;
//   }
  debugv( fprintf(stderr, "_make1dconf(%d): %d x %d\n", N, *gridSize, *blockSize) );
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

#ifdef __cplusplus
}
#endif
