#include "field.h"

#ifdef __cplusplus
extern "C" {
#endif


field_plan *new_fieldplan(param *params, tensor* kernel){
  field_plan* field = (field_plan*)malloc(sizeof(field_plan));
  field->params = params;  
  //field->convplan = new_gpuconv2(params->size, params->demagKernelSize, kernel); ///@todo for Arne.
  return field;
}


void field_evaluation(field_plan *plan, tensor *m, tensor *h){

  gpuconv2_exec(plan->convplan, m, h);

  float* hExt = plan->params->hExt;
  if(hExt[X] != 0. && hExt[Y] != 0. && hExt[Z] != 0.){
    //gpu_addZeeman(h, hExt);
  }

  if(plan->params->anisType != NONE){
    // ...
  }


  return;
}


#ifdef __cplusplus
}
#endif