#include "field.h"

#ifdef __cplusplus
extern "C" {
#endif


fieldplan *new_fieldplan(param *params, conv_data *conv){

  fieldplan* field = (fieldplan*)malloc(sizeof(fieldplan));
 
  field->params = params;
  field->conv = conv;

  return field;
}


void evaluate_field(fieldplan *plan, tensor *m, tensor *h){

  evaluate_convolution(m, h, plan->conv, plan->params);
//  cpu_addExch (m, h, plan->params);       //it is checked internally if exchange is already included in the convolution
//   add_exchange (m, h, plan->params);       //it is checked internally if exchange is already included in the convolution

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