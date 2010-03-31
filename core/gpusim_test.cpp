#include "gpusim.h"
#include "tensor.h"
#include <assert.h>
#include <stdio.h>

int main(int argc, char** argv){
  printf("gpusim_test\n");
  
  assert(argc == 4);
  
  FILE* mfile = fopen(argv[1], "rb");
  tensor* m = read_tensor(mfile);
  fclose(mfile);
  
  int N0 = m->size[1];
  int N1 = m->size[2];
  int N2 = m->size[3];
  printf("read m: %d x %d x %d\n", N0, N1, N2);
  
  FILE* kernelfile = fopen(argv[2], "rb");
  tensor* kernel = read_tensor(kernelfile);
  fclose(kernelfile);
  printf("read kernel: %d x %d x %d\n", kernel->size[2], kernel->size[3], kernel->size[4]);
  
  gpusim* sim = new_gpusim(N0, N1, N2, kernel);
  
  gpusim_loadm(sim, m);
  gpusim_updateh(sim);
  
  tensor* h = new_tensor(4, 3, N0, N1, N2);
  memcpy_from_gpu(sim->h, h->list, tensor_length(h));
  
  FILE* hfile = fopen(argv[3], "wb");
  write_tensor(h, hfile);
  fclose(hfile);
  
  //gpusim_storem(sim, m);
  //format_tensor(m, stdout);
  
  
  return 0;
}