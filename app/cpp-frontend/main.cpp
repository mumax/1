#include "tensor.h"
#include "gpuheun.h"
#include "timer.h"
#include "units.h"
#include <assert.h>
#include <stdio.h>

/**
 * params: m0 kernel msat aexch alpha dt steps savem [hext_x, hext_z, hext_y]
 */
int main(int argc, char** argv){
  
  assert(argc >= 8+1);
  
  //_______________________________________________________________ m0
  FILE* mfile = fopen(argv[1], "rb");
  tensor* m = read_tensor(mfile);
  fclose(mfile);
  
  //_______________________________________________________________ size
  int N0 = m->size[1];
  int N1 = m->size[2];
  int N2 = m->size[3];
  printf("read m: %d x %d x %d\n", N0, N1, N2);
  
  //_______________________________________________________________ kernel
  // todo: need safe_fopen
  FILE* kernelfile = fopen(argv[2], "rb");
  tensor* kernel = read_tensor(kernelfile);
  fclose(kernelfile);
  printf("read kernel: %d x %d x %d\n", kernel->size[2], kernel->size[3], kernel->size[4]);
  
  //_______________________________________________________________ msat, aexch, alpha
  units* unit = new_units();
  unit->msat = atof(argv[3]);
  unit->aexch = atof(argv[4]);
  float alpha = atof(argv[5]);
  
  //_______________________________________________________________ dt, steps, savem
  float dt = atof(argv[6]);
  int steps = atoi(argv[7]);
  int savem = atoi(argv[8]);
  
  //_______________________________________________________________ hExt
  float* hExt = new float[3];
  hExt[X] = hExt[Y] = hExt[Z] = 0;
  if(argc > 8+1){
    assert(argc == 8+3+1);
    hExt[X] = atof(argv[9])  / unitfield(unit);
    hExt[Y] = atof(argv[10]) / unitfield(unit);
    hExt[Z] = atof(argv[11]) / unitfield(unit);
  }
  
  //_______________________________________________________________ info
  
  printunits(unit);
  printf("initial m :\t%s\n", argv[1]);
  printf("kernel    :\t%s\n", argv[2]);
  printunits(unit);
  printf("alpha     :\t%g\n", alpha);
  printf("dt        :\t%E s\n", dt);
  printf("steps     :\t%d\n", steps);
  printf("save every:\t%d steps\n", savem);
  printf("ext. field:\t%g %g %g (T)\n", hExt[X], hExt[Y], hExt[Z]);
  
  fflush(stdout);
  
  //_______________________________________________________________ run
  
  gpuheun* heun = new_gpuheun(N0, N1, N2, kernel, hExt);
  gpuheun_loadm(heun, m);
  
  char* fname = (char*)calloc(257, sizeof(char));
  for(int i=0; i<steps; i++){
    if(i%savem == 0){
      //___________________________________________________________ save m
      gpuheun_storem(heun, m);
      sprintf(fname, "m%07d.t", i);
      FILE* file = fopen(fname, "wb");
      write_tensor(m, file);
      fclose(file);
    }
    gpuheun_step(heun, dt/unittime(unit), alpha);
  }
  //___________________________________________________________ time
  timer_printdetail();
  return 0;
}