#include "tensor.h"
#include "gpuheun.h"
#include "timer.h"
#include "pipes.h"
#include <assert.h>
#include <stdio.h>


int main(int argc, char** argv){
  printf("gpusim_test\n");
  
//   assert(argc == 3);
//   
//   FILE* mfile = fopen(argv[1], "rb");
//   tensor* m = read_tensor(mfile);
//   fclose(mfile);
//   
//   int N0 = m->size[1];
//   int N1 = m->size[2];
//   int N2 = m->size[3];
//   printf("read m: %d x %d x %d\n", N0, N1, N2);
  
  // todo: need safe_fopen
//   FILE* kernelfile = fopen(argv[2], "rb");
//   tensor* kernel = read_tensor(kernelfile);
//   fclose(kernelfile);

  tensor* kernel = pipe_tensor((char*)"kernel --size 64 64 8 --msat 800E3 --aexch 1.3e-11 --cellsize 1e-9 1e-9 1e-9");
  printf("read kernel: %d x %d x %d\n", kernel->size[2], kernel->size[3], kernel->size[4]);
  
//   // prblm 4:
// //   µ0Hx=-24.6 mT, µ0Hy= 4.3 mT, µ0Hz= 0.0 mT
//   
//   float* hExt = new float[3];
//   hExt[X] = 0.;//-24.6E-3; 
//   hExt[Y] = 0.;//4.3E-3;
//   hExt[Z] = 0;
//   
//   gpuheun* heun = new_gpuheun(N0, N1, N2, kernel, hExt);
//   
//   gpuheun_loadm(heun, m);
//   
//   char* fname = (char*)calloc(257, sizeof(char));
//   //printf("\27[H");
//   
//   for(int i=0; i<1000; i++){
//     printf("%d ", i);
//     fflush(stdout);
//     gpuheun_storem(heun, m);
//     sprintf(fname, "m%07d.t", i);
//     FILE* file = fopen(fname, "wb");
//     write_tensor(m, file);
//     fclose(file);
//     for(int j=0; j<1000; j++){
// 	gpuheun_step(heun, 1E-7, 1.0);
//     }
//   }
//   printf("\n");
//   timer_printdetail();

  return 0;
}