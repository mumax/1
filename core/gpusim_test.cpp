#include "tensor.h"
#include "gpurk4.h"
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
  
  gpurk4* rk4 = new_gpurk4(N0, N1, N2, kernel);
  
  gpurk4_loadm(rk4, m);
  
  char* fname = (char*)calloc(257, sizeof(char));
  for(int i=0; i<1000; i++){
    printf("%d ", i);
    fflush(stdout);
    gpurk4_storem(rk4, m);
    sprintf(fname, "m%07d.t", i);
    FILE* file = fopen(fname, "wb");
    write_tensor(m, file);
    fclose(file);
    for(int j=0; j<10; j++){
	gpurk4_step(rk4, 1E-5);
    }
  }

  return 0;
}