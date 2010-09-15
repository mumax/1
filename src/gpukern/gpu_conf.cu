#include "gpu_conf.h"
#include "gpu_properties.h"
#include "../macros.h"
#include <assert.h>

#ifdef __cplusplus
extern "C" {
#endif

void check3dconf(dim3 gridSize, dim3 blockSize){

  debugvv( printf("check3dconf((%d, %d, %d),(%d, %d, %d))\n", gridSize.x, gridSize.y, gridSize.z, blockSize.x, blockSize.y, blockSize.z) );
  
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


void make1dconf(int N, dim3* gridSize, dim3* blockSize){

  debugvv( printf("make1dconf(%d)\n", N) );
  
  cudaDeviceProp* prop = (cudaDeviceProp*)gpu_getproperties();
  int maxBlockSize = prop->maxThreadsPerBlock;
  if(maxBlockSize > 128){
    debugvv( fprintf(stderr, "WARNING: using 128 as max block size! \n") );
    maxBlockSize = 128;
  }
  int maxGridSize = prop->maxGridSize[X];

  (*blockSize).x = maxBlockSize;
  (*blockSize).y = 1;
  (*blockSize).z = 1;
  
  int N2 = divUp(N, maxBlockSize); // N2 blocks left
  
  int NX = divUp(N2, maxGridSize);
  int NY = divUp(N2, NX);

  (*gridSize).x = NX;
  (*gridSize).y = NY;
  (*gridSize).z = 1;

  assert((*gridSize).x * (*gridSize).y * (*gridSize).z * (*blockSize).x * (*blockSize).y * (*blockSize).z >= N);
  //assert((*gridSize).x * (*gridSize).y * (*gridSize).z * (*blockSize).x * (*blockSize).y * (*blockSize).z < N + maxBlockSize); ///@todo remove this assertion for very large problems
  
  check3dconf(*gridSize, *blockSize);
}

#ifdef __cplusplus
}
#endif
