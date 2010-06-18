#ifndef TIMESTEP_H
#define TIMESTEP_H

#include "tensor.h"
#include "param.h"
#include "field.h"
#include "gpueuler.h"
#include "gpuheun.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * A timestepper provides a easy-to-use way to make time steps.
 *
 * First, a timestepper is constructed with new_timestepper(param*, fieldplan*);
 * then, timestep(timestepper*, tensor *m, double *totalTime);
 * advances the state of m a bit in time, and updates the value of totalTime.
 *
 * The timestepper can internally use any kind of solver, determined
 * by param->solverType. It acts thus like an abstract class, providing
 * a transparent way to acces any solver without having to know which one.
 */
typedef struct{

  fieldplan* field;
  tensor* h;
  param* params;
  void* solver;
  
}timestepper;


timestepper *new_timestepper(param *, fieldplan* field);

void timestep(timestepper *ts, tensor *m, double *totalTime);


#ifdef __cplusplus
}
#endif
#endif