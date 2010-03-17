package sim

import(
  . "../tensor";
  . "math";
)

/**
 * The euler solver (time stepper) is used for testing, 
 * it's not very useful for real simulations.
 */
type Euler struct{
  m_tensor, h_tensor *Tensor4;
  m, h [][]float;	// components as lists
  field *FieldPlan;
  t, dt float;
}

func NewEuler(m *Tensor4, field *FieldPlan, dt float) *Euler{
  solver := new(Euler);
  
  solver.m_tensor = m;
  solver.h_tensor = NewTensor4(m.Size());
  // store m and h components as lists.
  solver.m, solver.h = make([][]float, 3), make([][]float, 3);
  for c:=0; c<3; c++{
      solver.m[c] = solver.m_tensor.Component(c).List();
      solver.h[c] = solver.h_tensor.Component(c).List();
  }

  solver.t = 0.;
  solver.dt = dt;
  solver.field = field;
  return solver;
}

func (s *Euler) Step(){
    alpha := 1.0;
    gilbert := 1.0 / (1.0 + alpha * alpha);
    
    s.field.Execute(s.m_tensor, s.h_tensor);
    m, h, torque := make([]float, 3), make([]float, 3), make([]float, 3);
    M := s.m;
    H := s.h;
    for i:=0; i<len(s.m[0]); i++{
      for c:=X; c<=Z; c++{
	m[c] = M[c][i];
	h[c] = H[c][i];
      }
      Torque(m, h, torque, alpha, gilbert);
      //Println(torque);
      norm:=0.;
      for c:=X; c<=Z; c++{
	m[c] += s.dt * torque[c];
	norm += m[c]*m[c];
      }
      norm = 1./float(Sqrt(float64(norm)));
      for c:=X; c<=Z; c++{
	M[c][i] = m[c] * norm;
      }
    }
}




