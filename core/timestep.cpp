#include "timestep.h"

#ifdef __cplusplus
extern "C" {
#endif


timestepper *new_timestepper(param *params, fieldplan* field){
  timestepper* ts = (timestepper*)malloc(sizeof(timestepper));
  ts->params = params;
  ts->field = field;
  ts->h = new_gputensor(4, tensor_size4D(params->size));
  
  int solverType = ts->params->solverType;
  
  if(solverType == SOLVER_EULER){
    ts->solver = new_gpueuler(params);
  }
  else if(solverType == SOLVER_HEUN){
    ts->solver = new_gpuheun(params);
  }
  else{
    fprintf(stderr, "Unknown solver type: %d\n", ts->params->solverType);
    abort();
  }

  return ts;
}


void timestep(timestepper *ts, tensor *m, double *totalTime){

  int solverType = ts->params->solverType;
  
  if(solverType == SOLVER_EULER){
    
    //gpueuler* euler = (gpueuler*)ts->solver;
    //gpueuler_step(euler, m, h, totalTime);
    
  }
  else if(solverType == SOLVER_HEUN){
    
    gpuheun* heun = (gpuheun*)ts->solver;
    gpuheun_step(heun, m, ts->h, totalTime);
    
  }
  else{
    fprintf(stderr, "Unknown solver type: %d\n", ts->params->solverType);
    abort();
  }
  
}





#ifdef __cplusplus
}
#endif