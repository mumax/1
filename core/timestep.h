#ifndef TIMESTEP_H
#define TIMESTEP_H

#include "tensor.h"
#include "param.h"
#include "field.h"
#include "gpueuler.h"
#include "gpuheun2.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct{

  fieldplan *field;
  param* params;
  void* solver;
  
}timestepper;



void timestep(timestepper *ts, tensor *m, tensor *h, double *total_time);


timestepper *new_timestepper(param *, fieldplan* field);

#ifdef __cplusplus
}
#endif
#endif