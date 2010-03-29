#include "euler.h"

eulersolver* new_euler(tensor* m, convplan* field, double dt){
  eulersolver* s = (eulersolver*)malloc(sizeof(eulersolver)); 
  
  s->m_tensor = m;
  s->h_tensor = new_tensorN(4, m->size);
  
  s->m = (float**)calloc(3, sizeof(float*));
  s->h = (float**)calloc(3, sizeof(float*));
  for(int i=0; i<3; i++){
    s->m[i] = tensor_component(s->m_tensor, i)->list;
    s->h[i] = tensor_component(s->h_tensor, i)->list;
  }
  
  s->size = (int*)calloc(3, sizeof(int));
  s->size[0] = m->size[1];
  s->size[1] = m->size[2];
  s->size[2] = m->size[3];
  s->N = s->size[0] * s->size[1] * s->size[2];
  printf("new_euler: %d x %d x %d = %d\n", s->size[0], s->size[1], s->size[2], s->N);
  
  s->torque_buf = (float*)calloc(3, sizeof(float));
  
  s->t = 0.;
  s->dt = dt;
  s->field = field;
  
  return s;
}

void euler_step(eulersolver* s){
     float** m = s->m;
     float** h = s->h;

    double alpha = 1.0;
    double gilbert = 1.0 / (1.0 + alpha * alpha);
    
    conv_execute(s->field, s->m_tensor->list, s->h_tensor->list);
    
    printf("s->N=%d\n", s->N);
    for(int i=0; i < s->N; i++){
      torque(m[X][i], m[Y][i], m[Z][i], h[X][i], h[Y][i], h[Z][i], alpha, gilbert, s->torque_buf);
      printf("torque = %f, %f, %f\n", s->torque_buf[X], s->torque_buf[Y], s->torque_buf[Z]);
      float norm  = 0.;
      for(int c=X; c<=Z; c++){
	m[c][i] += s->dt * s->torque_buf[c];
	norm += m[c][i]*m[c][i];
      }
    
      norm = 1./(sqrt(norm));
      for(int c=X; c<=Z; c++){
	m[c][i] *= norm;
      }
      
    }
    
    s->t += s->dt;
}