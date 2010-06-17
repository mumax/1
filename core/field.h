/**
 * @file
 *
 * 
 *
 * @author Arne Vansteenkiste, Ben Van de Wiele
 *
 */
#ifndef FIELD_H
#define FIELD_H

#include "tensor.h"
#include "param.h"
#include "gpuconv2.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct{

  param* params;
  gpuconv2* convplan;
  
}field_plan;



void field_evaluation(field_plan *plan, tensor *m, tensor *h);

field_plan *new_fieldplan(param *);

#ifdef __cplusplus
}
#endif
#endif