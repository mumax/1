#include "main.h"

int main(int argc, char** argv){
  
  gpu_override_stride(1);
  
  printf("*** Device properties ***\n");
  print_device_properties(stdout);

  param* p = read_param();
  printf("\n*** Simulation parameters ***\n");
  param_print(stdout, p);

  // this will be replaced by some routine that reads/creates an initial magnetization
  int* size4D = tensor_size4D(p->size);
  tensor* mHost = new_tensorN(4, size4D);
    for(int i=0; i<mHost->len; i++){
    mHost->list[i] = 1.;
  }
  write_tensor_fname(mHost, (char*)"m_init.t");
  
  tensor* m = new_gputensor(4, size4D);    //size4D puts a 3 in front of a size
  tensor_copy_to_gpu(mHost, m);
  
  
  // start of the actual simulation
  
  tensor* kernel = pipe_kernel(p);//init_kernel(p);
  fieldplan* field = new_fieldplan(p, kernel); 
  timestepper* ts = new_timestepper(p, field);  // allocates space for h internally

  double totalTime = 0.;
  
  for(int i=0; i<1000; i++){
    timestep(ts, m, &totalTime);
  }
  
  printf("\n*** Timing ***\n");
  timer_printdetail();
  return 0;
}


param* read_param(){

  // this will be replaced by a routine that reads a file or something.
  param* p = new_param();

  p->msat = 800E3;
  p->aexch = 1.3E-11;
  p->alpha = 1.0;

  p->size[X] = 1;
  p->size[Y] = 32;
  p->size[Z] = 128;

  double L = unitlength(p);
  p->cellSize[X] = (3E-9 / p->size[X]) / L;
  p->cellSize[Y] = (50E-9 / p->size[Y]) / L;
  p->cellSize[Z] = (500E-9 / p->size[Z]) / L;

  p->demagCoarse[X] = 1;
  p->demagCoarse[Y] = 1;
  p->demagCoarse[Z] = 1;
  
  p->demagPeriodic[X] = 0;
  p->demagPeriodic[Y] = 0;
  p->demagPeriodic[Z] = 0;

  for (int i=0; i<3; i++){
    p->kernelSize[i] = 2 * p->size[i]; 
  }

  
  p->kernelType = KERNEL_MICROMAG3D;
  p->solverType = SOLVER_HEUN;

  double T = unittime(p);
  p->maxDt = 0.1E-12 / T;
  
  check_param(p);
  return p;

}
