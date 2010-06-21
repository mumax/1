#include "field.h"

#ifdef __cplusplus
extern "C" {
#endif


fieldplan *new_fieldplan(param *params, tensor* kernel){
  fieldplan* field = (fieldplan*)malloc(sizeof(fieldplan));
  field->params = params;  
  field->convplan = new_gpuconv2(params->size, params->kernelSize);
  gpuconv2_loadkernel5DSymm(field->convplan, kernel);
  return field;
}


void field_evaluation(fieldplan *plan, tensor *m, tensor *h){

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