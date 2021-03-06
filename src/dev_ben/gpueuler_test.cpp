/**
 * @file
 * This test program runs a small simulation using gpuconv1 and gpueuler.
 * The initial magnetization and demag tensor are read from testm0.t and testkernel.t.
 * A few time steps are taken and one spin of the result is compared with its known solution.
 *
 * @author Arne Vansteenkiste
 *
 */
#include "tensor.h"
#include "gpueuler.h"
#include <assert.h>
#include <stdio.h>

int main(int argc, char** argv){
  printf("gpueuler_test\n");
  
  assert(argc == 3);
  
  FILE* mfile = fopen(argv[1], "rb");
  tensor* m = read_tensor(mfile);
  fclose(mfile);
  
  int N0 = m->size[1];
  int N1 = m->size[2];
  int N2 = m->size[3];
  printf("read m: %d x %d x %d\n", N0, N1, N2);
  
  // todo: need safe_fopen
  FILE* kernelfile = fopen(argv[2], "rb");
  tensor* kernel = read_tensor(kernelfile);
  fclose(kernelfile);
  printf("read kernel: %d x %d x %d\n", kernel->size[2], kernel->size[3], kernel->size[4]);
  
  gpueuler* euler = new_gpueuler(N0, N1, N2, kernel);
  
  gpueuler_loadm(euler, m);
  
  for(int i=0; i<100; i++){
    gpueuler_step(euler, 1E-7);
  }
  gpueuler_storem(euler, m);
  
  //assert(fabs(*tensor_get(m, 4, 0, 0, 0, 0) - 0.776904) < 1E-6);
  printf("%f\nPASS\n", *tensor_get(m, 4, 0, 0, 0, 0));

  return 0;
}