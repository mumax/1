#include "tensor.h"
#include "gpuanal1.h"
#include "timer.h"
#include "units.h"
#include "pipes.h"
#include <assert.h>
#include <stdio.h>

/**
 * params: m0 kernel msat aexch alpha dt steps savem [hext_x, hext_z, hext_y]
 */
int main(int argc, char** argv){
  
  int a = 1;			// skip program name
  char* m0 = argv[a++];		// m0 file also determines mesh size
  float cellsizeX = atof(argv[a++]);
  float cellsizeY = atof(argv[a++]);
  float cellsizeZ = atof(argv[a++]);
  float msat = atof(argv[a++]);
  float aexch = atof(argv[a++]);
  float alpha = atof(argv[a++]);
  float dt = atof(argv[a++]);
  int steps = atoi(argv[a++]);
  int savem = atoi(argv[a++]);
  float hx = 0.;
  float hy = 0.;
  float hz = 0.;
  
  if(argc >= a){
    hx = atof(argv[a++]);
    hy = atof(argv[a++]);
    hz = atof(argv[a++]);
  }
  
  //_______________________________________________________________ m0
  FILE* mfile = fopen(m0, "rb");
  tensor* m = read_tensor(mfile);
  fclose(mfile);
  
  //_______________________________________________________________ size
  int N0 = m->size[1];	// size[0] is 3, for the X,Y,Z components of the vector field
  int N1 = m->size[2];
  int N2 = m->size[3];
  printf("read m: %d x %d x %d\n", N0, N1, N2);
  
  //_______________________________________________________________ kernel
  // todo: need safe_fopen
  char* command = new char[4000];
  sprintf(command, "kernel --size %d %d %d --cellsize %g %g %g --msat %g --aexch %g\n" , N0, N1, N2, cellsizeX, cellsizeY, cellsizeZ, msat, aexch);
  fprintf(stderr, "%s", command);
  tensor* kernel = pipe_tensor(command);
  printf("read kernel: %d x %d x %d\n", kernel->size[2], kernel->size[3], kernel->size[4]);
  
  //_______________________________________________________________ msat, aexch, alpha
  units* unit = new_units();
  unit->msat = msat;
  unit->aexch = aexch;
  
  
  //_______________________________________________________________ hExt
 
  float* hExt = new float[3];
  hExt[X] = hx  / unitfield(unit);
  hExt[Y] = hy / unitfield(unit);
  hExt[Z] = hz / unitfield(unit);
  
  //_______________________________________________________________ info
  
  printunits(unit);
  printf("initial m :\t%s\n", argv[1]);
  printf("kernel    :\t%s\n", argv[2]);
  printunits(unit);
  printf("alpha     :\t%g\n", alpha);
  printf("dt        :\t%E s\n", dt);
  printf("steps     :\t%d\n", steps);
  printf("save every:\t%d steps\n", savem);
  printf("ext. field:\t%g %g %g (Msat)\n", hExt[X], hExt[Y], hExt[Z]);
  
  fflush(stdout);
  
  //_______________________________________________________________ run
  
  gpuanal1* heun = new_gpuanal1(N0, N1, N2, kernel);
  gpuanal1_loadm(heun, m);
  
  char* fname = (char*)calloc(257, sizeof(char));
  for(int i=0; i<steps; i++){
    if(i%savem == 0){
      //___________________________________________________________ save m
      gpuanal1_storem(heun, m);
      sprintf(fname, "m%07d.t", i);
      FILE* file = fopen(fname, "wb");
      write_tensor(m, file);
      fclose(file);
    }
    gpuanal1_step(heun, dt/unittime(unit));
  }
  //___________________________________________________________ time
  timer_printdetail();
  return 0;
}