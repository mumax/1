#ifndef EULER_H
#define EULER_H

// allow inclusion in C++ code
#ifdef __cplusplus
extern "C" {
#endif

#include "solver.h"

typedef struct{
  
  tensor* m_tensor;
  tensor* h_tensor;
  float** m;
  float** h;
  
  int* size;
  int N;
  
  convplan* field;
  
  float* torque_buf;
  
  float t;
  float dt;
  
}eulersolver;

eulersolver* new_euler(tensor* m, convplan* field, double dt);

void euler_step(eulersolver* solver);

#ifdef __cplusplus
}
#endif

#endif