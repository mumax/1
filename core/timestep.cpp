#include "timestep.h"

#ifdef __cplusplus
extern "C" {
#endif


void timestep(timestepper *ts, tensor *m, tensor *h, double *total_time){

  int solverType = ts->params->solverType;
  if(solverType == SOLVER_EULER){
    gpueuler* euler = (gpueuler*)ts->solver;
    //gpueuler_step(euler, m, h, total_time);
  }
  else if(solverType == SOLVER_HEUN){
    gpuheun2* heun = (gpuheun2*)ts->solver;
    gpuheun2_step(heun, m, h, total_time);
  }
  else{
    fprintf(stderr, "Unknown solver type: %d\n", ts->params->solverType);
    abort();
  }
  
}


timestepper *new_timestepper(param *params, fieldplan* field){
  timestepper* ts = (timestepper*)malloc(sizeof(timestepper));
  ts->params = params;
  ts->field = field;
  
  int solverType = ts->params->solverType;
  if(solverType == SOLVER_EULER){
    //ts->solver = new_gpueuler(params);
  }
  else if(solverType == SOLVER_HEUN){
    //ts->solver = new_gpuheun(params);
  }
  else{
    fprintf(stderr, "Unknown solver type: %d\n", ts->params->solverType);
    abort();
  }

  return ts;
}


#ifdef __cplusplus
}
#endif