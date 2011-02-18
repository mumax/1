#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "cpu_kernel_micromag2d.h"
#include "cpu_kernel_micromag3d.h"
#include "thread_functions.h"
#include "../macros.h"

int main(int argc, char** argv){

//  int co1 = atoi(argv[1]);
//  int co2 = atoi(argv[2]);

  fprintf(stderr, "Kernel initialization: CPU\n");
  if(argc != 11){
    fprintf(stderr, "Kernel initialization needs 11 command-line arguments.\n");
	abort();
  }
  int kernelSize[3];
  kernelSize[0] = atoi(argv[1]);
  kernelSize[1] = atoi(argv[2]);
  kernelSize[2] = atoi(argv[3]);
  
  float cellSize[3];
  int kernelType = -1;
  if (strcmp(argv[4], "inf") == 0){
    cellSize[0] = -1.0f;
    kernelType = 2;                  //micromag2D kernel
  }
  else{
    cellSize[0] = atof(argv[4]);
    kernelType = 3;                  //micromag2D kernel
  }
  cellSize[1] = atof(argv[5]);
  cellSize[2] = atof(argv[6]);

  int repetition[3];
  repetition[0] = atoi(argv[7]);
  repetition[1] = atoi(argv[8]);
  repetition[2] = atoi(argv[9]);
  
  int Nthreads = atoi(argv[10]);
  
  init_Threads(Nthreads);

  if (kernelType==-1){
    fprintf(stderr, "In kernel initialization: wrong cell size: kernel type could not be recognized\n");
	abort();
  }

  // x[i],y[i] loops over XX, YY, ZZ, YZ, XZ, XY
  int x[6] = {X, Y, Z, Y, X, X};
  int y[6] = {X, Y, Z, Z, Z, Y};
  for(int i=0; i<6; i++){
    if (kernelType==2)
      cpu_init_kernel_elements_micromag2d(x[i], y[i], kernelSize, cellSize, repetition);
      
    if (kernelType==3)
      cpu_init_kernel_elements_micromag3d(x[i], y[i], kernelSize, cellSize, repetition);
  }
  return (0);
}
