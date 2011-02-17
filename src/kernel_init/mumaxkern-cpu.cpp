#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include "cpu_kernel_micromag2d.h"
#include "cpu_kernel_micromag3d.h"


int main(int argc, char** argv){

  int co1 = atoi(argv[1]);
  int co2 = atoi(argv[2]);

  int kernelSize[3];
  kernelSize[0] = atoi(argv[3]);
  kernelSize[1] = atoi(argv[4]);
  kernelSize[2] = atoi(argv[5]);
  
  float cellSize[3];
  int kernelType = -1;
  if (strcmp(argv[6], "inf") == 0){
    cellSize[0] = -1.0f;
    kernelType = 2;                  //micromag2D kernel
  }
  else{
    cellSize[0] = atof(argv[6]);
    kernelType = 3;                  //micromag2D kernel
  }
  cellSize[1] = atof(argv[7]);
  cellSize[2] = atof(argv[8]);

  int repetition[3];
  repetition[0] = atoi(argv[9]);
  repetition[1] = atoi(argv[10]);
  repetition[2] = atoi(argv[11]);
  
  int Nthreads = atoi(argv[12]);
  
  init_Threads(int Nthreads);

  if (kernelType==-1)
    fprintf(stderr, "In kernel initialization: wrong cell size: kernel type could not be recognized\n");

  if (kernelType==2)
    cpu_init_kernel_elements_micromag2d(co1, co2, kernelSize, cellSize, repetition);
    
  if (kernelType==3)
    cpu_init_kernel_elements_micromag3d(co1, co2, kernelSize, cellSize, repetition);

  return (0);
}
