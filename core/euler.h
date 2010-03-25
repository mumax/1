#ifndef EULER_H
#define EULER_H

#include "solver.h"

typedef struct{
  tensor* m_tensor;
  tensor* h_tensor;
  float** m;
  float** h;
  convplan* field;
  float t;
  float dt;
}eulersolver;

eulersolver* new_euler(tensor* m, convplan* field);

void euler_step(eulersolver* solver);

#endif