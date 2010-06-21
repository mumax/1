/**
 * @file
 * A timestepper provides a easy-to-use way to make time steps.
 *
 * First, a timestepper is constructed with new_timestepper(param*, fieldplan*);
 * then, timestep(timestepper*, tensor *m, double *totalTime);
 * advances the state of m a bit in time, and updates the value of totalTime.
 *
 * The timestepper can internally use any kind of solver, determined
 * by param->solverType. It acts thus like an abstract class, providing
 * a transparent way to acces any solver without having to know which one.
 *
 * @author Ben Van de Wiele, Arne Vansteenkiste
 */

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


typedef struct{

  fieldplan* field;
  tensor* h;
  param* params;
  void* solver;
  
}timestepper;

/**
 * Creates a new timestepper
 */
timestepper *new_timestepper(param* params,         ///< The type of solver and its parameters are taken from here
                             fieldplan* field       ///< Plan used to update the effective field
                             );

/**
 *  Takes one full time step
 */
void timestep(timestepper *ts,                      ///< timestepper to used
              tensor *m,                            ///< magnetization to advance in time
              double *totalTime                     ///< starts with the time at the beginning of the step, is updated to totalTime + deltaT by the stepper
              );


#ifdef __cplusplus
}
#endif
#endif