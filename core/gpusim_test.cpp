#include "gpusim.h"
#include "tensor.h"
#include <assert.h>
#include <stdio.h>

int main(int argc, char** argv){
  printf("gpusim_test\n");
  
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
  
  gpusim* sim = new_gpusim(N0, N1, N2, kernel);
  
  gpusim_loadm(sim, m);
  
  char* fname = (char*)calloc(257, sizeof(char));
  for(int i=0; i<1000; i++){
    printf("%d ", i);
    fflush(stdout);
    gpusim_storem(sim, m);
    sprintf(fname, "m%07d.t", i);
    FILE* file = fopen(fname, "wb");
    write_tensor(m, file);
    fclose(file);
    for(int j=0; j<10; j++){
	gpusim_eulerstep(sim, 0.00001);
    }
  }

  return 0;
}