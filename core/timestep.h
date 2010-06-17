#ifndef TIMESTEP_H
#define TIMESTEP_H

#include "tensor.h"
#include "param.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct{

  field_plan *field;
  param* params;
  void* solver;
  
}timestepper;



void timestep(timestepper *ts, tensor *m, tensor *h, double *total_time);


timestepper *new_timestepper(param *);

#ifdef __cplusplus
}
#endif
#endif