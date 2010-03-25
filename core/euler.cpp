#include "euler.h"

eulersolver* new_euler(tensor* m, convplan* field, double dt){
  eulersolver* s = (eulersolver*)malloc(sizeof(eulersolver));
  
  s->m_tensor = m;
  s->h_tensor = new_tensor(4, m->size);

  s->t = 0.;
  s->dt = dt;
  s->field = field;
  
  return s;
}

void euler_step(eulersolver* s){
    double alpha = 1.0;
    double gilbert = 1.0 / (1.0 + alpha * alpha);
    
    conv_execute(s->field, s->m_tensor->list, s->h_tensor->list);
//     m, h, torque := make([]float, 3), make([]float, 3), make([]float, 3);
//     M := s.m;
//     H := s.h;
//     for i:=0; i<len(s.m[0]); i++{
//       for c:=X; c<=Z; c++{
// 	m[c] = M[c][i];
// 	h[c] = H[c][i];
//       }
//       Torque(m, h, torque, alpha, gilbert);
//       //Println(torque);
//       norm:=0.;
//       for c:=X; c<=Z; c++{
// 	m[c] += s.dt * torque[c];
// 	norm += m[c]*m[c];
//       }
//       norm = 1./float(Sqrt(float64(norm)));
//       for c:=X; c<=Z; c++{
// 	M[c][i] = m[c] * norm;
//       }
//     }
}