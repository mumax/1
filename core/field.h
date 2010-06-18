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
  
}fieldplan;



void field_evaluation(fieldplan *plan, tensor *m, tensor *h);

fieldplan *new_fieldplan(param *, tensor* kernel);

#ifdef __cplusplus
}
#endif
#endif