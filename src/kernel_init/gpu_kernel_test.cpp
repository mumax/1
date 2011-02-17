#include "../macros.h"
#include "cpu_kernel_micromag3d.h"
#include "gpu_kernel_micromag3d.h"
#include "cpu_kernel_micromag2d.h"
#include "gpu_kernel_micromag2d.h"

int main(int argc, char** argv){
  
  int kernelSize[3], exchInConv[3], repetition[3];
  float cellSize[3];
  
  kernelSize[X] = 2*4;
  kernelSize[Y] = 2*4;
  kernelSize[Z] = 2*4;
  
  cellSize[X] = 1.0f;
  cellSize[Y] = 1.0f;
  cellSize[Z] = 1.0f;
  
  exchInConv[X] = 1;
  exchInConv[Y] = 1;
  exchInConv[Z] = 1;
  
  repetition[X] = 0;
  repetition[Y] = 0;
  repetition[Z] = 0;
  
  int exchType = EXCH_6NGBR;
//   int exchType = EXCH_12NGBR;
  
  int kernelType = KERNEL_MICROMAG3D;
//   int kernelType = KERNEL_MICROMAG2D;
  

  cpu_kernel_micromag3d(int *kernelSize, float *cellSize, int exchType, int *exchInConv, int *repetition);
//   gpu_kernel_micromag3d(int *kernelSize, float *cellSize, int exchType, int *exchInConv, int *repetition);
//   cpu_kernel_micromag2d(int *kernelSize, float *cellsize, int exchType, int *exchInConv, int *repetition);
//   gpu_kernel_micromag2d(int *kernelSize, float *cellsize, int exchType, int *exchInConv, int *repetition);

  return 0;
}


