#ifndef TIMESTEP_H
#define TIMESTEP_H

#include "tensor.h"
#include "param.h"
#include "field.h"
#include "gpueuler.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct{

  field_plan *field;
  param* params;
  void* solver;
  
}timestepper;



void timestep(timestepper *ts, tensor *m, tensor *h, double *total_time);


timestepper *new_timestepper(param *, field_plan* field);

#ifdef __cplusplus
}
#endif
#endif