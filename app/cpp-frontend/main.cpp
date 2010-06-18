#include "main.h"

int main(int argc, char** argv){
  
  printf("*** Device properties ***\n");
  print_device_properties(stdout);

  param* p = read_param();
  printf("\n*** Simulation parameters ***\n");
  param_print(stdout, p);

  
  int* size4D = tensor_size4D(p->size);

  tensor* mHost = new_tensorN(4, size4D);
    for(int i=0; i<mHost->len; i++){
    mHost->list[i] = 1.;
  }
  
  tensor* m = new_gputensor(4, size4D);    //size4D puts a 3 in front of a size
  tensor_copy_to_gpu(mHost, m);
  
  tensor* kernel = init_kernel(p);
  fieldplan* field = new_fieldplan(p, kernel);
  timestepper* ts = new_timestepper(p, field);

  double totalTime = 0.;
  
  for(int i=0; i<1000; i++){
    timestep(ts, m, &totalTime);
  }
  
  printf("\n*** Timing ***\n");
  timer_printdetail();
  return 0;
}


param* read_param(){
  
  param* p = new_param();

  p->msat = 800E3;
  p->aexch = 1.1E-13;
  p->alpha = 1.0;

  p->size[X] = 1;
  p->size[Y] = 32;
  p->size[Z] = 128;

  double L = unitlength(p);
  p->cellSize[X] = 1E-9 / L;
  p->cellSize[Y] = 1E-9 / L;
  p->cellSize[Z] = 1E-9 / L;

  p->kernelType = KERNEL_MICROMAG3D;
  p->kernelSize[X] = 2*p->size[X];
  p->kernelSize[Y] = 2*p->size[Y];
  p->kernelSize[Z] = 2*p->size[Z];

  p->solverType = SOLVER_HEUN;

  return p;

}