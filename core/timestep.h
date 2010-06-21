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
#include "gpunormalize.h"
#include "gpueuler.h"
#include "gpuheun.h"

#ifdef __cplusplus
extern "C" {
#endif

///@internal
typedef struct{

  fieldplan* field;     ///< plan to update h, called each time before a solver is asked to step.
  tensor* h;            ///< stores the effective field. The user does not need to worry about allocating it etc.
  param* params;
  int totalSteps;       ///< total number of time steps (stages actually) taken. Used to normalize m once every "normalizeEvery" steps
  void* solver;         ///< can point to many types of solvers. params->solverType tells which (e.g.: euler, heun, ...)
  
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