#include "gpusim.h"
#include "tensor.h"

#include <stdio.h>

int main(int argc, char** argv){
  printf("gpusim_test\n");
  
  FILE* mfile = fopen(argv[1], "rb");
  tensor* m = read_tensor(mfile);
  fclose(mfile);
  printf("read m: %d x %d x %d\n", m->size[1], m->size[2], m->size[3]);

  FILE* kernelfile = fopen(argv[2], "rb");
  tensor* kernel = read_tensor(kernelfile);
  fclose(kernelfile);
  printf("read kernel: %d x %d x %d\n", kernel->size[2], kernel->size[3], kernel->size[4]);
  
  gpusim* sim = new_gpusim(m->size[1], m->size[2], m->size[3], kernel);
  
  gpusim_loadm(sim, m);
  gpusim_storem(sim, m);
  
  format_tensor(m, stdout);
  
  return 0;
}