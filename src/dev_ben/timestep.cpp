#include "timestep.h"

#ifdef __cplusplus
extern "C" {
#endif


void timestep(timestepper *ts, tensor *m, double *totalTime){

  switch (ts->params->solverType){
/*    case SOLVER_EULER:
      evaluate_euler_step (ts, m, totalTime);
      break;*/
    case SOLVER_HEUN:
      evaluate_heun_step (ts, m, totalTime);
      break;
    case SOLVER_ANAL_FW:
      evaluate_anal_fw_step (ts, m, totalTime);
      break;
    case SOLVER_ANAL_PC:
      evaluate_anal_pc_step (ts, m, totalTime);
      break;
    default:
       fprintf(stderr, "abort: no valid solverType %d\n", ts->params->solverType);
      abort();
  }
  
  return;
}


void evaluate_heun_step(timestepper *ts, tensor *m, double *totalTime){

  gpuheun* heun = (gpuheun*)ts->solver;
  for(int i=0; i<2; i++){                     // heun has two stages
    evaluate_field(ts->field, m, ts->h);
    gpu_heun_step(heun, m, ts->h, totalTime);  // difference between stage 0 and 1 is taken care of internally by heun
    ts->totalSteps++;
    if(ts->totalSteps % ts->params->normalizeEvery == 0){
      gpu_normalize_uniform(m->list, m->len/3);
    }
  }
  
  return;
}


void evaluate_anal_fw_step(timestepper *ts, tensor *m, double *totalTime){

  gpuanalfw* anal_fw = (gpuanalfw*)ts->solver;
  evaluate_field(ts->field, m, ts->h);

  gpu_anal_fw_step(anal_fw->params, m, m, ts->h);

  if (ts->totalSteps % 10 == 0)              // to correct rounding of errors
    gpu_normalize_uniform(m->list, m->len/3);
  
  ts->totalSteps++;
  *totalTime += ts->params->maxDt;

  return;
}

void evaluate_anal_pc_step(timestepper *ts, tensor *m, double *totalTime){

  gpuanalpc* anal_pc = (gpuanalpc*)ts->solver;

  evaluate_field(ts->field, m, ts->h);
  gpu_anal_fw_step(anal_pc->params, m, anal_pc->m2, ts->h);
  evaluate_field(ts->field, anal_pc->m2, anal_pc->h2);
  gpu_anal_pc_mean_h(ts->h, anal_pc->h2);
  gpu_anal_fw_step(anal_pc->params, m, m, ts->h);

  if (ts->totalSteps % 10 == 0)             // to correct rounding of errors
    gpu_normalize_uniform(m->list, m->len/3);

  *totalTime += ts->params->maxDt;
  ts->totalSteps++;

  
  return;
}


// void evaluate_euler_step(timestepper *ts, tensor *m, double *totalTime){
// 
//   gpueuler* euler = (gpueuler*)ts->solver;
// 
//   // ....
//   
//   return;
// }



timestepper *new_timestepper(param* p, fieldplan* field){
  
  timestepper* ts = (timestepper*)malloc(sizeof(timestepper));
  ts->params = p;
  ts->field = field;
  ts->h = new_gputensor(4, tensor_size4D(p->size));
  ts->totalSteps = 0;
  ts->solver = NULL;
  
  int solverType = ts->params->solverType;
  
  switch (ts->params->solverType){
/*    case SOLVER_EULER:
      ts->solver = new_gpueuler(p);
      break;*/
    case SOLVER_HEUN:
      ts->solver = new_gpuheun(p);
      break;
    case SOLVER_ANAL_FW:
      ts->solver = new_gpuanalfw(p);
      break;
    case SOLVER_ANAL_PC:
      ts->solver = new_gpuanalpc(p);
      break;
    default:
       fprintf(stderr, "abort: no valid solverType %d\n", ts->params->solverType);
      abort();
  }

  
/*  if(solverType == SOLVER_EULER){
    ts->solver = new_gpueuler(p);
  }
  else*/ 
  if(solverType == SOLVER_HEUN){
    ts->solver = new_gpuheun(p);
  }
  else if(solverType == SOLVER_ANAL_FW){
    ts->solver = new_gpuanalfw(p);
  }
  else if(solverType == SOLVER_ANAL_PC){
    ts->solver = new_gpuanalpc(p);
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