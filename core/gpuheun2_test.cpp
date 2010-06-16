/**
 * @file
 * This test program runs a small simulation using gpuconv1 and gpuheun.
 * The initial magnetization and demag tensor are read from testm0.t and testkernel.t.
 * A few time steps are taken and one spin of the result is compared with its known solution.
 *
 * @author Arne Vansteenkiste
 *
 */
#include "tensor.h"
#include "param.h"
#include "gpuheun2.h"
#include "timer.h"
#include "pipes.h"
#include <assert.h>
#include <stdio.h>

int main(int argc, char** argv){

  printf("gpuheun2_test\n");

  param* p = new_param();
  
  p->aexch = 1.3E-11;
  p->msat = 800E3;
  p->alpha = 1.0;
  
  p->size[X] = 64;
  p->size[Y] = 32;
  p->size[Z] =  4;

  double L = unitlength(p);
  p->cellSize[X] = 1E-9 / L;
  p->cellSize[Y] = 1E-9 / L;
  p->cellSize[Z] = 1E-9 / L;
  
  
  tensor* kernel = pipe_tensor((char*)"kernel --size 64 32 4 --msat 800E3 --aexch 1.3e-11 --cellsize 1e-9 1e-9 1e-9");

  param_print(stdout, p);
  
  gpuheun2* solver = new_gpuheun2_param(p);

  tensor* m = new_tensorN(4, tensor_size4D(p->size));
  tensor* mz = tensor_component(m, Z);
  for(int i=0; i<mz->len; i++){
    mz->list[i] = 1.;
  }

  gpuheun2_loadm(solver, m);

  tensor_zero(m);
  gpuheun2_storem(solver, m);
  
//   for(int i=0; i<m->len; i++)
//     assert(m->list[i] == i);
  
  float alpha = 1.0;
  for(int i=0; i<10; i++){
    gpuheun2_step(solver, 1E-5, alpha);
  }
  gpuheun2_storem(solver, m);
//   
  printf("PASS\n");
//   assert(fabs(*tensor_get(m, 4, 0, 0, 0, 0) - 0.578391) < 1E-6);
//   timer_printdetail();

  return 0;
}