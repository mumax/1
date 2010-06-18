#include "gpu_micromag3d_kernel.h"
#include "gputil.h"
#include "kernel.h"
#include "field.h"
#include "timestep.h"
#include "param.h"

param* read_param();
void check_param(param *p);


int main(int argc, char** argv){
  
  printf("*** Device properties ***\n");
  print_device_properties(stdout);

  param* p = read_param();
  printf("\n*** Simulation parameters ***\n");
  param_print(stdout, p);

  
  tensor* kernel = init_kernel(p);


  return 0;
}


param* read_param(){
  
  param* p = new_param();

  p->msat = 800E3;
  p->aexch = 1.1E-13;
  p->alpha = 1.0;

  p->size[X] = 4;
  p->size[Y] = 4;
  p->size[Z] = 8;

  double L = unitlength(p);
  p->cellSize[X] = 1E-9 / L;
  p->cellSize[Y] = 1E-9 / L;
  p->cellSize[Z] = 1E-9 / L;

  p->demagCoarse[X] = 1;
  p->demagCoarse[Y] = 1;
  p->demagCoarse[Z] = 1;
  
  p->demagPeriodic[X] = 0;
  p->demagPeriodic[Y] = 0;
  p->demagPeriodic[Z] = 0;

  int zero_pad[3];
  for (int i=0; i<3; i++){
    zero_pad[i] = (!p->demagPeriodic[i]) ? 1:0;
    p->kernelSize[i] = (1 + zero_pad[i]) * p->size[i]/p->demagCoarse[i]; 
  }
  if (p->size[Xeventueel]==1) 
    p->kernelSize[X] = 1;
  
  p->kernelType = KERNEL_MICROMAG3D;
  
  return p;

}
